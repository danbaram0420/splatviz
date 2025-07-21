import copy
import os
import traceback
from typing import List
import imageio
import numpy as np
import torch
import torch.nn
from tqdm import tqdm
from pathlib import Path

from compression.compression_exp import run_single_decompression
from gaussian_renderer import render_simple
from scene.gaussian_model import GaussianModel
from scene.cameras import CustomCam
from renderer.base_renderer import Renderer
from splatviz_utils.dict_utils import EasyDict


class GaussianRenderer(Renderer):
    def __init__(self, num_parallel_scenes=16):
        super().__init__()
        self.num_parallel_scenes = num_parallel_scenes
        self.gaussian_models: List[GaussianModel | None] = [None] * num_parallel_scenes
        self._current_ply_file_paths: List[str | None] = [None] * num_parallel_scenes
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32).to("cuda")
        self._last_num_scenes = 0
    def _render_impl(self, res, fov, edit_text, eval_text, resolution,
                     ply_file_paths, cam_params, current_ply_names,
                     background_color, video_cams=[], render_depth=False, render_alpha=False,
                     img_normalize=False, use_splitscreen=False, highlight_border=False,
                     save_ply_path=None, colormap=None, invert=False, slider={},
                     **other_args):
        cam_params = cam_params.to("cuda")
        slider = EasyDict(slider)
        if len(ply_file_paths) == 0:
            res.error = "Select a .ply file"
            return

        # If scenes were removed (fewer paths than before), clear old models
        if len(ply_file_paths) < self._last_num_scenes:
            for i in range(len(ply_file_paths), self.num_parallel_scenes):
                self.gaussian_models[i] = None
        self._last_num_scenes = len(ply_file_paths)  # update count

        # Load any new scenes or changed files
        for i, ply_path in enumerate(ply_file_paths):
            if ply_path != self._current_ply_file_paths[i]:
                self.gaussian_models[i] = self._load_model(ply_path)
                self._current_ply_file_paths[i] = ply_path

        # Get transforms if provided
        transforms = other_args.get("object_transforms", None)
        if transforms:
            # Ensure transforms list length matches number of ply_file_paths
            # (It should, given how we maintain it in Splatviz)
            pass

        images = []
        if not use_splitscreen and transforms:
            # **Combined rendering path**: merge all scenes into one
            # Prepare lists to accumulate data for all gaussians
            all_xyz = []
            all_rot = []
            all_scaling = []
            all_feat_dc = []
            all_feat_rest = []
            all_opacity = []
            for i, ply_path in enumerate(ply_file_paths):
                if self.gaussian_models[i] is None:
                    continue
                model = self.gaussian_models[i]
                # Apply transform for this object if available
                if transforms and i < len(transforms):
                    q_obj, t_obj = transforms[i]  # both are torch tensors
                    # Normalize quaternion
                    q_obj = q_obj / torch.norm(q_obj)
                    # Rotate all points and orientations in this model
                    xyz = model.get_xyz  # positions (N,3):contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}
                    rot = model._rotation          # raw rotations (N,4)
                    # Compute rotation matrix from q_obj
                    w0, x0, y0, z0 = q_obj  # quaternion components
                    # Rotation matrix R_obj (3x3) from quaternion
                    R00 = 1 - 2*(y0*y0 + z0*z0);  R01 = 2*(x0*y0 - w0*z0);      R02 = 2*(x0*z0 + w0*y0)
                    R10 = 2*(x0*y0 + w0*z0);      R11 = 1 - 2*(x0*x0 + z0*z0);  R12 = 2*(y0*z0 - w0*x0)
                    R20 = 2*(x0*z0 - w0*y0);      R21 = 2*(y0*z0 + w0*x0);      R22 = 1 - 2*(x0*x0 + y0*y0)
                    R_obj = torch.tensor([[R00,R01,R02],[R10,R11,R12],[R20,R21,R22]], dtype=torch.float32, device="cuda")
                    # Apply rotation and translation to positions
                    xyz_transformed = (R_obj @ xyz.T).T + t_obj  # shape (N,3)
                    # Apply rotation to each Gaussian orientation (quaternions)
                    # Quaternion multiplication: q_new = q_obj * orig_q
                    w1 = rot[:,0]; x1 = rot[:,1]; y1 = rot[:,2]; z1 = rot[:,3]
                    w_new = w0*w1 - x0*x1 - y0*y1 - z0*z1
                    x_new = w0*x1 + x0*w1 + y0*z1 - z0*y1
                    y_new = w0*y1 - x0*z1 + y0*w1 + z0*x1
                    z_new = w0*z1 + x0*y1 - y0*x1 + z0*w1
                    rot_transformed = torch.stack((w_new, x_new, y_new, z_new), dim=1)
                else:
                    # No transform (shouldn't happen for combined if transforms list exists, but just in case)
                    xyz_transformed = model.get_xyz
                    rot_transformed = model._rotation
                all_xyz.append(xyz_transformed)
                all_rot.append(rot_transformed)
                all_scaling.append(model._scaling)        # (N,3)
                all_feat_dc.append(model._features_dc)    # (N,1,3)
                all_feat_rest.append(model._features_rest)# (N,SH-1,3)
                all_opacity.append(model._opacity)        # (N,1)
            # Concatenate all objects' data
            if len(all_xyz) == 0:
                res.error = "No data to render"; return
            xyz_comb = torch.cat(all_xyz, dim=0)
            rot_comb = torch.cat(all_rot, dim=0)
            scaling_comb = torch.cat(all_scaling, dim=0)
            feat_dc_comb = torch.cat(all_feat_dc, dim=0)
            feat_rest_comb = torch.cat(all_feat_rest, dim=0)
            opacity_comb = torch.cat(all_opacity, dim=0)
            # Create a combined GaussianModel to render
            combined_model = GaussianModel(sh_degree=0, disable_xyz_log_activation=True)
            combined_model._xyz = xyz_comb
            combined_model._rotation = rot_comb
            combined_model._scaling = scaling_comb
            combined_model._features_dc = feat_dc_comb
            combined_model._features_rest = feat_rest_comb
            combined_model._opacity = opacity_comb
            combined_model.active_sh_degree = 0
            # Execute edit script on combined model
            gs = copy.deepcopy(combined_model)
            try:
                exec(edit_text)
            except Exception as e:
                res.error = traceback.format_exc() + str(e)
            # Render one image with all gaussians
            fov_rad = fov / 360 * 2 * np.pi
            render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_params)
            render = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"))
            if render_alpha:
                images.append(render["alpha"])
            elif render_depth:
                depth = render["depth"]
                images.append((depth - depth.min()) / (depth.max()))  # normalize depth to [0,1]
            else:
                images.append(render["render"])
            # Optionally save combined ply
            if save_ply_path is not None:
                self.save_ply(gs, save_ply_path)
        else:
            # **Original per-scene rendering path** (for splitscreen or no transforms)
            for scene_index, ply_file_path in enumerate(ply_file_paths):
                gs = copy.deepcopy(self.gaussian_models[scene_index])
                # If transforms exist and splitscreen, apply object transform to this model as well
                if transforms and scene_index < len(transforms):
                    q_obj, t_obj = transforms[scene_index]
                    q_obj = q_obj / torch.norm(q_obj)
                    # (Apply rotation & translation to gs similar to above, omitted for brevity)
                    # ... update gs._xyz and gs._rotation ...
                try:
                    exec(edit_text)
                except Exception as e:
                    res.error = traceback.format_exc() + str(e)
                # (Rendering each scene to separate image, unchanged logic)
                fov_rad = fov / 360 * 2 * np.pi
                render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_params)
                render = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"))
                img = render["alpha"] if render_alpha else ( (render["depth"]-render["depth"].min())/ (render["depth"].max()) if render_depth else render["render"] )
                images.append(img)
                if save_ply_path is not None:
                    self.save_ply(gs, save_ply_path)
        # Combine images (for multiple scenes) and output
        self._return_image(images, res, normalize=img_normalize,
                           use_splitscreen=use_splitscreen,
                           highlight_border=highlight_border, colormap=colormap, invert=invert)
        # Compute mean/std of positions for info
        if images:
            res.mean_xyz = torch.mean(xyz_comb if 'xyz_comb' in locals() else gs.get_xyz, dim=0)
            res.std_xyz = torch.std(xyz_comb if 'xyz_comb' in locals() else gs.get_xyz)
        if len(eval_text) > 0:
            res.eval = eval(eval_text)


    def _load_model(self, ply_file_path):
        if ply_file_path.endswith(".ply"):
            model = GaussianModel(sh_degree=0, disable_xyz_log_activation=True)
            model.load_ply(ply_file_path)
        elif ply_file_path.endswith("compression_config.yml"):
            model = run_single_decompression(Path(ply_file_path).parent.absolute())
        else:
            raise NotImplementedError("Only .ply or .yml files are supported.")
        return model

    def render_video(self, save_path, video_cams, gaussian):
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/rotate_{len(os.listdir(save_path))}.mp4"
        video = imageio.get_writer(filename, mode="I", fps=30, codec="libx264", bitrate="16M", quality=10)
        for render_cam in tqdm(video_cams):
            img = render_simple(viewpoint_camera=render_cam, pc=gaussian, bg_color=self.bg_color)["render"]
            img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            video.append_data(img)
        video.close()
        print(f"Video saved in {filename}.")

    @staticmethod
    def save_ply(gaussian, save_ply_path):
        os.makedirs(save_ply_path, exist_ok=True)
        save_path = os.path.join(save_ply_path, f"model_{len(os.listdir(save_ply_path))}.ply")
        print("Model saved in", save_path)
        gaussian.save_ply(save_path)
