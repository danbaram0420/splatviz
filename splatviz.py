from imgui_bundle import imgui
import numpy as np
import torch
import sys
import os
import pybullet as p, trimesh
from trimesh.transformations import quaternion_from_matrix

sys.path.append("./gaussian-splatting")
torch.set_printoptions(precision=2, sci_mode=False)
np.set_printoptions(precision=2)

from scipy.spatial.transform import Rotation as R

from renderer.renderer_wrapper import RendererWrapper
from renderer.gaussian_renderer import GaussianRenderer
from renderer.gan_renderer import GANRenderer

from renderer.attach_renderer import AttachRenderer
from splatviz_utils.gui_utils import imgui_window
from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils import gl_utils
from splatviz_utils.gui_utils import text_utils
from splatviz_utils.gui_utils.constants import *
from splatviz_utils.dict_utils import EasyDict
from widgets import (
    edit,
    eval,
    performance,
    load_pkl,
    load_ply,
    camera,
    save,
    latent,
    render,
    training,
)

class Splatviz(imgui_window.ImguiWindow):
    def __init__(self, data_path, mode, host, port, quat, gan_path="", scene_path="", objects_path="", ):
        self.code_font_path = "resources/fonts/jetbrainsmono/JetBrainsMono-Regular.ttf"
        self.regular_font_path = "resources/fonts/source_sans_pro/SourceSansPro-Regular.otf"

        super().__init__(
            title="splatviz",
            window_width=1920,
            window_height=1080,
            font=self.regular_font_path,
            code_font=self.code_font_path,
        )

        self.code_font = imgui.get_io().fonts.add_font_from_file_ttf(self.code_font_path, 14)
        self.regular_font = imgui.get_io().fonts.add_font_from_file_ttf(self.code_font_path, 14)
        self._imgui_renderer.refresh_font_texture()

        # Internals.
        self._last_error_print = None

        self.quat = quat

        # Determine initial files to load
        initial_files = None
        if scene_path:
            initial_files = []
            # Always include the scene .ply first
            initial_files.append(os.path.abspath(scene_path))
            # Include all .ply files from the objects directory
            if objects_path:
                for fname in os.listdir(objects_path):
                    if fname.endswith(".ply"):
                        initial_files.append(os.path.abspath(os.path.join(objects_path, fname)))
            # Sort object files for consistent order (scene is index 0, objects 1..N)
            # (If a specific order is needed, the user can name files accordingly)
            initial_files[1:] = sorted(initial_files[1:])

        self.widgets = []
        update_all_the_time = True
        if mode == "default":
            # Pass initial_files to LoadWidget if provided
            self.widgets = [
                load_ply.LoadWidget(self, data_path, initial_files=initial_files),
                camera.CamWidget(self),
                performance.PerformanceWidget(self),
                save.CaptureWidget(self),
                render.RenderWidget(self),
                edit.EditWidget(self),
                eval.EvalWidget(self),
            ]
            renderer = GaussianRenderer()
        elif mode == "attach":
            self.widgets = [
                camera.CamWidget(self),
                performance.PerformanceWidget(self),
                render.RenderWidget(self),
                edit.EditWidget(self),
                training.TrainingWidget(self),
            ]
            sys.path.append(gan_path)
            renderer = AttachRenderer(host=host, port=port)
            update_all_the_time = True
        elif mode == "gan":
            self.widgets = [
                load_pkl.LoadWidget(self, data_path, file_ending=".pkl"),
                camera.CamWidget(self, fov=12, radius=2.7, up_direction=1),
                performance.PerformanceWidget(self),
                save.CaptureWidget(self),
                render.RenderWidget(self),
                edit.EditWidget(self),
                eval.EvalWidget(self),
                latent.LatentWidget(self),
            ]
            sys.path.append(gan_path)
            renderer = GANRenderer()
        else:
            raise NotImplementedError(f"Mode '{mode}' not recognized.")

        self.renderer = RendererWrapper(renderer, update_all_the_time)
        self._tex_img = None
        self._tex_obj = None

        # Widget interface.
        self.args = EasyDict()
        self.result = EasyDict()
        self.eval_result = ""

        # After initializing widgets and renderer...
        # Initialize transform list for each loaded scene/object
        # Each transform is stored as (quat, trans), quat=[1,0,0,0] (w,x,y,z), trans=[0,0,0] initially
        num_initial = len(self.widgets[0].plys)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        identity_trans = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        self.physics_initials = []
        self.object_transforms = [(identity_quat, identity_trans) for _ in range(num_initial)]

        for idx, ply_path in enumerate(initial_files):
            if idx == 0:
                self.load_scene_physics(ply_path, self.quat)
            else:
                self.load_dynamic_object(ply_path, self.quat, initial=True)

        self.pending_object_path = None
        self.fixed_depth = 3.0

        # Initialize window.
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame()

    def close(self):
        for widget in self.widgets:
            widget.close()
        super().close()

    def print_error(self, error):
        error = str(error)
        if error != self._last_error_print:
            print(f"\n{error}\n")
            self._last_error_print = error

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(min(self.content_width / 120, self.content_height / 60))
        if self.font_size != old:
            self.skip_frame()

    def _set_sizes(self):
        self.pane_w = max(self.content_width - self.content_height, 500)
        self.button_w = self.font_size * 5
        self.button_large_w = self.font_size * 10
        self.label_w = round(self.font_size * 5.5) + 100
        self.label_w_large = round(self.font_size * 5.5) + 150

    def set_transform(self, object_idx: int, quat: tuple[float, float, float, float],
                      trans: tuple[float, float, float]) -> None:
        """Update the rotation (quat) and translation for a given object index."""
        if object_idx < 0 or object_idx >= len(self.object_transforms):
            print(f"set_transform: invalid object index {object_idx}")
            return
        # Convert quaternion and translation to torch tensors (on GPU)
        # Expect quat in (w,x,y,z) format
        q = torch.tensor(quat, dtype=torch.float32, device="cuda")
        t = torch.tensor(trans, dtype=torch.float32, device="cuda")
        # Normalize quaternion to be safe
        q = q / torch.norm(q)
        self.object_transforms[object_idx] = (q, t)

    def draw_frame(self):
        self.begin_frame()
        self.args = EasyDict()
        self._set_sizes()

        # Control pane
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_size(imgui.ImVec2(self.pane_w, self.content_height))
        control_pane_flags = WINDOW_NO_TITLE_BAR | WINDOW_NO_RESIZE | WINDOW_NO_MOVE
        imgui.begin("##control_pane", p_open=True, flags=control_pane_flags)

        # Widgets
        for widget in self.widgets:
            expanded, _visible = imgui_utils.collapsing_header(widget.name, default=widget.name == "Load")
            imgui.indent()
            widget(expanded)
            imgui.unindent()

        # imgui.show_style_editor()

        # **Sync transform list with current number of objects**
        current_files = self.widgets[0].plys
        # Build a map of old file->transform
        prev_map = {}
        if hasattr(self, "object_transforms"):
            prev_files = getattr(self.widgets[0], "prev_plys", None) or current_files
            for j, fname in enumerate(prev_files):
                if j < len(self.object_transforms):
                    prev_map[fname] = self.object_transforms[j]
        # Update prev_plys for next frame
        self.widgets[0].prev_plys = list(current_files)
        # Rebuild object_transforms for current list
        new_transforms = []
        for fname in current_files:
            if fname in prev_map:
                new_transforms.append(prev_map[fname])
            else:
                # New file (added) -> default identity transform
                new_transforms.append((identity_quat.clone(), identity_trans.clone()))
        self.object_transforms = new_transforms

        # Pass transforms to renderer via args
        self.args.object_transforms = self.object_transforms

        # Render
        if self.is_skipping_frames():
            pass
        else:
            self.renderer.set_args(**self.args)
            result = self.renderer.result
            if result is not None:
                self.result = result

        # Display
        max_w = self.content_width - self.pane_w
        max_h = self.content_height
        pos = np.array([self.pane_w + max_w / 2, max_h / 2])
        if "image" in self.result:
            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                    self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
                else:
                    self._tex_obj.update(self._tex_img)
            zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
            self._tex_obj.draw(pos=pos, zoom=zoom, align=0.5, rint=True)
        if "error" in self.result:
            self.print_error(self.result.error)
            if "message" not in self.result:
                self.result.message = str(self.result.error)
        if "message" in self.result:
            tex = text_utils.get_texture(
                self.result.message,
                size=self.font_size,
                max_width=max_w,
                max_height=max_h,
                outline=2,
            )
            tex.draw(pos=pos, align=0.5, rint=True, color=1)
        if "eval" in self.result:
            self.eval_result = self.result.eval
        else:
            self.eval_result = None

        if self.pending_object_path and imgui.is_mouse_clicked(imgui.MouseButton_.left):
            mx, my = imgui.get_mouse_pos()
            vw, vh = self.content_width, self.content_height
            nx = (mx / vw) * 2 - 1
            ny = -(my / vh) * 2 + 1

            cam_pos, cam_dir = self.widgets[1].get_camera_ray(nx, ny)
            target_pos = cam_pos + cam_dir * self.fixed_depth
            self.load_dynamic_object(self.pending_object_path, target_pos, initial=False)
            self.pending_object_path = None

        # End frame.
        self._adjust_font_size()
        imgui.end()
        self.end_frame()

    def load_scene_physics(self, ply_path, quat):
        #quat는 wxyz form으로 가정
        folder = os.path.dirname(ply_path)
        cand = [f for f in os.listdir(folder) if f.lower().endswith(".obj")]
        if not cand:
            print("No .obj found next to", ply_path)
            return
        obj_path = os.path.join(folder, cand[0])
        sc_col = p.createCollisionShape(p.GEOM_MESH,
                                        fileName=obj_path,
                                        flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        sc_vis = p.createVisualShape(p.GEOM_MESH, fileName=obj_path)
        scene_uid = p.createMultiBody(baseMass=0,
                                      baseCollisionShapeIndex=sc_col,
                                      baseVisualShapeIndex=sc_vis, basePosition=[0, 0, 0], baseOrientation=quat)
        print("scene_uid =", scene_uid, " sc_vis =", sc_vis)  # -1 이면 로드 실패
        assert scene_uid >= 0 and sc_vis >= 0, "scene mesh 로드 실패!"
        self.physics_initials.append((scene_uid,quat, [0, 0, 0]))

    def load_dynamic_object(self, ply_path, pos, initial=False):
        scene_quat, _ = self.object_transforms[0]
        if not initial:
            self.widgets[0].plys.append(os.path.abspath(ply_path))
            gaussian_t = torch.tensor(pos, dtype=torch.float32, device="cuda")
            gaussian_q = torch.tensor([1,0,0,0], dtype=torch.float32, device="cuda")
            self.object_transforms.append((gaussian_q, gaussian_t))

        folder = os.path.dirname(ply_path)
        cand = [f for f in os.listdir(folder) if f.lower().endswith(".obj")]
        if not cand:
            print("No .obj found next to", ply_path)
            return
        obj_path = os.path.join(folder, cand[0])
        if initial:
            physics_pos = [0,0,0]
        else:
            physics_pos = R.from_quat(scene_quat).apply(pos)
        self.bullet_load_obj(obj_path, physics_pos, scene_quat.tolist())

    def bullet_load_obj(self, obj_path, pos, quat):
        vhacd_path = obj_path.replace(".obj", "_vhacd.obj")
        if not os.path.exists(vhacd_path):
            p.vhacd(obj_path, vhacd_path, "vhacd_log.txt",concavity=0.002, resolution=1_000_000)
        mesh = trimesh.load(vhacd_path, force='mesh')
        com = mesh.center_mass
        col = p.createCollisionShape(p.GEOM_MESH, fileName=vhacd_path)
        vis = p.createVisualShape(p.GEOM_MESH, fileName=vhacd_path)
        I_tensor = mesh.moment_inertia
        eigval, eigvec = np.linalg.eigh(I_tensor)
        quat_I = quaternion_from_matrix(np.vstack([np.hstack([eigvec, [[0], [0], [0]]]),
                                                   [0, 0, 0, 1]])[:3, :3]).tolist()
        obj_uid = p.createMultiBody(baseMass=2.0,
                                    baseCollisionShapeIndex=col,
                                    baseVisualShapeIndex=vis,
                                    basePosition=[pos[0], pos[1], pos[2]],
                                    baseOrientation=quat,
                                    baseInertialFramePosition=com.tolist(),
                                    baseInertialFrameOrientation=quat_I)

        p.changeDynamics(obj_uid, -1,
                         lateralFriction=0.9,
                         rollingFriction=0.03, spinningFriction=0.03,
                         linearDamping=0.05, angularDamping=0.05,
                         frictionAnchor=1,
                         activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)
        self.physics_initials.append((obj_uid,quat_I,com.tolist()))
        print("obj created")
