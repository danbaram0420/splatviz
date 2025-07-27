from imgui_bundle import imgui
import torch
import numpy as np

from splatviz_utils.gui_utils.easy_imgui import label, slider, checkbox
from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.dict_utils import EasyDict
from splatviz_utils.cam_utils import (
    get_forward_vector,
    create_cam2world_matrix,
    get_origin,
    normalize_vecs,
)
from widgets.widget import Widget


class CamWidget(Widget):
    def __init__(self, viz, fov=60, radius=1, up_direction=-1, device="cuda"):
        super().__init__(viz, "Camera")
        self.device = device

        # cam params
        self.fov = fov
        self.radius = radius
        self.lookat_point = torch.tensor((0.0, 0.0, 0.0), device=device)
        self.cam_pos = torch.tensor([0.0, 0.0, -1.0], device=device)
        self.up_vector = torch.tensor([0.0, up_direction, 0.0], device=device)
        self.forward = torch.tensor([0.0, 0.0, 1.0], device=device)

        # controls
        self.pose = EasyDict(yaw=np.pi, pitch=0)
        self.invert_x = False
        self.invert_y = False
        self.move_speed = 0.02
        self.wasd_move_speed = 0.1
        self.drag_speed = 0.005
        self.rotate_speed = 0.002
        self.control_modes = ["Orbit", "WASD"]
        self.current_control_mode = 0
        self.last_drag_delta = imgui.ImVec2(0, 0)

        # momentum
        self.momentum_x = 0.0
        self.momentum_y = 0.0
        self.momentum_dropoff = 0.8
        self.momentum = 0.3

        self.locked_by_external = False

    @imgui_utils.scoped_by_object_id
    def __call__(self, show: bool):
        viz = self.viz
        active_region = EasyDict(x=viz.pane_w, y=0, width=viz.content_width - viz.pane_w, height=viz.content_height)
        self.handle_dragging_in_window(**active_region)
        self.handle_mouse_wheel()
        self.handle_wasd()

        if show:

            imgui.text("Camera Controls")
            label("Camera Mode", viz.label_w)
            _, self.current_control_mode = imgui.combo("##cam_modes", self.current_control_mode, self.control_modes)

            if self.control_modes[self.current_control_mode] == "WASD":
                label("Move Speed", viz.label_w)
                self.wasd_move_speed = slider(self.wasd_move_speed, "move_speed", 0.001, 1, log=True)

            label("Drag Speed", viz.label_w)
            self.drag_speed = slider(self.drag_speed, "drag_speed", 0.001, 0.1, log=True)

            label("Momentum", viz.label_w)
            self.momentum = slider(self.momentum, "momentum", 0.0, 0.999)

            label("Momentum drop-off", viz.label_w)
            self.momentum_dropoff = slider(self.momentum_dropoff, "momentum_dropoff", 0.0, 1.0)

            label("Rotate Speed", viz.label_w)
            self.rotate_speed = slider(self.rotate_speed, "rot_speed", 0.001, 0.1, log=True)

            label("Invert X", viz.label_w)
            self.invert_x = checkbox(self.invert_x, "invert_x")

            label("Invert Y", viz.label_w)
            self.invert_y = checkbox(self.invert_y, "invert_y")

            imgui.text("\nCamera Matrix")

            imgui.push_item_width(200)
            label("Up Vector", viz.label_w)
            _changed, up_vector_tuple = imgui.input_float3("##up_vector", v=self.up_vector.tolist(), format="%.1f")
            if _changed:
                self.up_vector = torch.tensor(up_vector_tuple, device=self.device)
            imgui.same_line()
            if imgui_utils.button("Set current direction", width=viz.button_large_w):
                self.up_vector = -self.forward
                self.pose.yaw = 0
                self.pose.pitch = 0
            imgui.same_line()
            if imgui_utils.button("Flip", width=viz.button_w):
                self.up_vector = -self.up_vector

            label("FOV", viz.label_w)
            self.fov = slider(self.fov, "##fov", 1, 180, format="%.2f °")
            imgui.same_line()
            changed, self.fov = imgui.input_float("##fov_input", self.fov)

            if self.control_modes[self.current_control_mode] == "Orbit":
                label("Camera Pos (yaw, pitch)", viz.label_w)
                _, (self.pose.yaw, self.pose.pitch) = imgui.input_float2("##yaw_ptich", [self.pose.yaw, self.pose.pitch], format="%.1f")

                label("Radius", viz.label_w)
                self.radius = slider(self.radius, "##radius", 1, 20, format="%.2f")
                imgui.same_line()
                changed, self.radius = imgui.input_float("##radius_input", self.radius)

                imgui.same_line()
                if imgui_utils.button("Set to xyz stddev", width=viz.button_large_w) and "std_xyz" in viz.result.keys():
                    self.radius = viz.result.std_xyz.item()

                label("Look at Point", viz.label_w)
                _, look_at_point_tuple = imgui.input_float3("##lookat", self.lookat_point.tolist(), format="%.1f")
                self.lookat_point = torch.tensor(look_at_point_tuple, device=self.device)
                imgui.same_line()
                if imgui_utils.button("Set to xyz mean", width=viz.button_large_w) and "mean_xyz" in viz.result.keys():
                    self.lookat_point = viz.result.mean_xyz
            imgui.pop_item_width()

        if not self.locked_by_external:
            self.cam_params = create_cam2world_matrix(
                self.forward, self.cam_pos, self.up_vector
            )[0]
        else:
            # 사용자가 마우스·키보드로 움직이면 잠금 해제
            if imgui.is_mouse_dragging(0) or len(self.viz.current_pressed_keys) > 0:
                self.locked_by_external = False
        if show:
            imgui.text("\nExtrinsics Matrix")
            imgui.input_float4("##extr0", self.cam_params.cpu().numpy().tolist()[0])
            imgui.input_float4("##extr1", self.cam_params.cpu().numpy().tolist()[1])
            imgui.input_float4("##extr2", self.cam_params.cpu().numpy().tolist()[2])
            imgui.input_float4("##extr3", self.cam_params.cpu().numpy().tolist()[3])

        viz.args.yaw = self.pose.yaw
        viz.args.pitch = self.pose.pitch
        viz.args.fov = self.fov
        viz.args.cam_params = self.cam_params

        # params for the video widget
        viz.args.lookat_point = self.lookat_point
        viz.args.up_vector = self.up_vector

    def handle_dragging_in_window(self, x, y, width, height):
        x_dir = -1 if self.invert_x else 1
        y_dir = -1 if self.invert_y else 1

        if imgui.is_mouse_dragging(0):  # left mouse button
            new_delta = imgui.get_mouse_drag_delta(0)
            if imgui_utils.did_drag_start_in_window(x, y, width, height, new_delta):
                delta = new_delta - self.last_drag_delta
                self.last_drag_delta = new_delta
                self.momentum_x = x_dir * delta.x * self.rotate_speed * (1 - self.momentum) + (self.momentum_x * self.momentum)
                self.momentum_y = y_dir * delta.y * self.rotate_speed * (1 - self.momentum) + (self.momentum_y * self.momentum)

        elif imgui.is_mouse_dragging(2) or imgui.is_mouse_dragging(1):  # right mouse button or middle mouse button
            new_delta = imgui.get_mouse_drag_delta(2)
            if imgui_utils.did_drag_start_in_window(x, y, width, height, new_delta):
                delta = new_delta - self.last_drag_delta
                self.last_drag_delta = new_delta

                right = torch.linalg.cross(self.forward, self.up_vector)
                right = right / torch.linalg.norm(right)
                cam_up = torch.linalg.cross(right, self.forward)
                cam_up = cam_up / torch.linalg.norm(cam_up)

                x_change = x_dir * right * -delta.x * self.drag_speed
                y_change = y_dir * cam_up * delta.y * self.drag_speed
                self.cam_pos += x_change
                self.cam_pos += y_change
                if self.control_modes[self.current_control_mode] == "Orbit":
                    self.lookat_point += x_change
                    self.lookat_point += y_change
        else:
            self.last_drag_delta = imgui.ImVec2(0, 0)

        self.pose.yaw += self.momentum_x
        self.pose.pitch += self.momentum_y
        self.momentum_x *= self.momentum_dropoff
        self.momentum_y *= self.momentum_dropoff
        self.pose.pitch = np.clip(self.pose.pitch, -np.pi / 2, np.pi / 2)

    def handle_wasd(self):
        if self.control_modes[self.current_control_mode] == "WASD":
            self.forward = get_forward_vector(
                lookat_position=self.cam_pos,
                horizontal_mean=self.pose.yaw + np.pi / 2,
                vertical_mean=self.pose.pitch + np.pi / 2,
                radius=0.01,
                up_vector=self.up_vector,
            )
            self.sideways = torch.linalg.cross(self.forward, self.up_vector)
            if imgui.is_key_down(imgui.Key.up_arrow) or "w" in self.viz.current_pressed_keys:
                self.cam_pos += self.forward * self.wasd_move_speed
            if imgui.is_key_down(imgui.Key.left_arrow) or "a" in self.viz.current_pressed_keys:
                self.cam_pos -= self.sideways * self.wasd_move_speed
            if imgui.is_key_down(imgui.Key.down_arrow) or "s" in self.viz.current_pressed_keys:
                self.cam_pos -= self.forward * self.wasd_move_speed
            if imgui.is_key_down(imgui.Key.right_arrow) or "d" in self.viz.current_pressed_keys:
                self.cam_pos += self.sideways * self.wasd_move_speed
            if "q" in self.viz.current_pressed_keys:
                self.cam_pos += self.up_vector * self.wasd_move_speed
            if "e" in self.viz.current_pressed_keys:
                self.cam_pos -= self.up_vector * self.wasd_move_speed

        elif self.control_modes[self.current_control_mode] == "Orbit":
            self.cam_pos = get_origin(
                self.pose.yaw + np.pi / 2,
                self.pose.pitch + np.pi / 2,
                self.radius,
                self.lookat_point,
                up_vector=self.up_vector,
            )
            self.forward = normalize_vecs(self.lookat_point - self.cam_pos)
            if imgui.is_key_down(imgui.Key.up_arrow) or "w" in self.viz.current_pressed_keys:
                self.pose.pitch += self.move_speed
            if imgui.is_key_down(imgui.Key.left_arrow) or "a" in self.viz.current_pressed_keys:
                self.pose.yaw += self.move_speed
            if imgui.is_key_down(imgui.Key.down_arrow) or "s" in self.viz.current_pressed_keys:
                self.pose.pitch -= self.move_speed
            if imgui.is_key_down(imgui.Key.right_arrow) or "d" in self.viz.current_pressed_keys:
                self.pose.yaw -= self.move_speed

    def handle_mouse_wheel(self):
        mouse_pos = imgui.get_io().mouse_pos
        if mouse_pos.x >= self.viz.pane_w:
            wheel = imgui.get_io().mouse_wheel
            if self.control_modes[self.current_control_mode] == "WASD":
                self.cam_pos += self.forward * self.move_speed * wheel
            elif self.control_modes[self.current_control_mode] == "Orbit":
                self.radius -= wheel / 10

    def set_external_camera_pose(self, matrix):
        mat = torch.as_tensor(matrix, dtype=torch.float32, device=self.device)

        # 행렬을 그대로 보존
        self.cam_params = mat
        self.locked_by_external = True  # 잠금

        # 내부 상태도 맞춰 둬야 WASD 등 조작이 정상
        self.cam_pos = mat[:3, 3]
        self.forward = mat[:3, 2] / torch.linalg.norm(mat[:3, 2])
        self.up_vector = mat[:3, 1] / torch.linalg.norm(mat[:3, 1])


    # def set_external_camera_pose(self, matrix):
    #     viz = self.viz
    #     world_ref = torch.as_tensor([0., 0., -1.], dtype=torch.float32, device=self.device)
    #     """4×4 camera-to-world 행렬을 그대로 적용 (yaw/pitch로 쪼개지 않음)."""
    #     mat = torch.as_tensor(matrix, dtype=torch.float32, device=self.device)
    #     print("set")
    #     print(mat)
    #
    #     # # 위치
    #     self.cam_pos = mat[:3, 3]
    #     # 카메라 지역축(OpenGL 기준) → 월드축
    #     forward = mat[:3, 2]  # +Z 가 시선 반대방향(카메라 앞쪽) ⇒ 그대로 사용
    #     # print(forward)
    #     up_vec = mat[:3, 1]  # +Y 가 '상'
    #     # print(up_vec)
    #     #
    #     # # 정규화
    #     forward = forward / torch.linalg.norm(forward)
    #     up_vec = up_vec / torch.linalg.norm(up_vec)
    #     #
    #     self.forward = forward
    #     self.up_vector = up_vec  # ★ 기존 고정값 갱신
    #
    #     yaw = torch.arctan2(forward[0], forward[2])
    #     pitch = torch.arctan2(forward[1], torch.sqrt(forward[0]**2 + forward[2]**2))
    #     yaw = yaw.cpu().numpy()
    #     pitch = pitch.cpu().numpy()
    #
    #     # pitch = torch.arcsin(torch.dot(forward, up_vec))
    #     # pitch = pitch.cpu().numpy()
    #     #
    #     # # horizontal forward
    #     # f_h = forward - torch.dot(forward, up_vec) * up_vec
    #     # f_h = f_h / torch.linalg.norm(f_h)
    #     #
    #     # # reference axis in the horizontal plane
    #     # ref = world_ref - torch.dot(world_ref, up_vec) * up_vec
    #     # if torch.linalg.norm(ref) < 1e-6:
    #     #     ref = torch.array([1., 0., 0.]) - torch.dot(torch.array([1., 0., 0.]), up_vec) * up_vec
    #     # ref = ref / torch.linalg.norm(ref)
    #     #
    #     # # yaw (+: left-turn around up)
    #     # yaw = torch.arctan2(
    #     #     torch.dot(torch.cross(ref, f_h), up_vec),
    #     #     torch.dot(ref, f_h)
    #     # )
    #     #
    #     # yaw = yaw.cpu().numpy()
    #
    #     self.pose.pitch = float(pitch)
    #     self.pose.yaw = float(yaw)
    #     self.current_control_mode = 1  # “WASD” 모드 고정
    #     #
    #     # # radius 유지(Orbit 전환 대비)
    #     self.radius = torch.linalg.norm(self.cam_pos - self.lookat_point).item()
    #     if self.radius == 0:
    #         self.radius = 1.0