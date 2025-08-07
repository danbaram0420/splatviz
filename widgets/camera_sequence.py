from imgui_bundle import imgui
import numpy as np
import math
import torch

from splatviz_utils.dict_utils import EasyDict
from splatviz_utils.cam_utils import create_cam2world_matrix
from splatviz_utils.gui_utils import imgui_utils
from widgets.widget import Widget
from widgets import camera
import time

import torch

class CameraSequenceWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Camera Sequence")
        # 저장된 카메라 포즈 리스트
        self.saved_cameras = []            # 각 요소는 EasyDict로 포즈 정보 저장
        # 인접한 카메라 간 전환 설정 (시간, 프레임 수)
        self.transition_times = []
        self.transition_frames = []
        # 재생 상태 관리
        self.playing = False
        self.paused = False
        self.current_segment = 0
        self.step_count = 0
        # 보간 계산을 위한 캐시 변수
        self.q_start = None
        self.q_end = None
        self.pos_start = None
        self.pos_end = None
        self.current_frames = 0

    def _get_quaternion_from_matrix(self, R: np.ndarray):
        """3x3 회전 행렬을 사원수(quaternion) (w,x,y,z)로 변환"""
        q = np.zeros(4)
        t = np.trace(R)
        if t > 0.0:
            t = math.sqrt(t + 1.0)
            q[0] = 0.5 * t
            t = 0.5 / t
            q[1] = (R[2,1] - R[1,2]) * t
            q[2] = (R[0,2] - R[2,0]) * t
            q[3] = (R[1,0] - R[0,1]) * t
        else:
            i = 0
            if R[1,1] > R[0,0]:
                i = 1
            if R[2,2] > R[i,i]:
                i = 2
            j = (i + 1) % 3
            k = (i + 2) % 3
            t = math.sqrt(R[i,i] - R[j,j] - R[k,k] + 1.0)
            q[i+1] = 0.5 * t
            t = 0.5 / t
            q[0] = (R[k,j] - R[j,k]) * t
            q[j+1] = (R[j,i] + R[i,j]) * t
            q[k+1] = (R[k,i] + R[i,k]) * t
        q /= np.linalg.norm(q)
        return q

    def _get_rotation_matrix_from_quaternion(self, q: np.ndarray):
        """사원수 (w,x,y,z)로부터 3x3 회전 행렬 생성"""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
            [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
            [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y]
        ], dtype=np.float64)

    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text("Save and Playback Camera Poses")
            # 현재 카메라 저장 버튼
            if imgui_utils.button("Save Current Camera", width=viz.button_w):
                cam_widget = None
                for w in viz.widgets:
                    if isinstance(w, camera.CamWidget):
                        cam_widget = w
                        break
                if cam_widget is not None:
                    cam = EasyDict()
                    mode_name = cam_widget.control_modes[cam_widget.current_control_mode]
                    cam.mode = mode_name
                    cam.yaw = float(cam_widget.pose.yaw)
                    cam.pitch = float(cam_widget.pose.pitch)
                    cam.up_vector = cam_widget.up_vector.clone() if hasattr(cam_widget.up_vector, "clone") else torch.tensor(cam_widget.up_vector)
                    cam.cam_pos = cam_widget.cam_pos.clone() if hasattr(cam_widget.cam_pos, "clone") else torch.tensor(cam_widget.cam_pos)
                    cam.forward = cam_widget.forward.clone() if hasattr(cam_widget.forward, "clone") else torch.tensor(cam_widget.forward)
                    cam.lookat_point = cam_widget.lookat_point.clone() if hasattr(cam_widget.lookat_point, "clone") else torch.tensor(cam_widget.lookat_point)
                    cam.radius = float(cam_widget.radius)
                    cam.cam_params = cam_widget.cam_params.clone() if hasattr(cam_widget.cam_params, "clone") else torch.tensor(cam_widget.cam_params)
                    cam.name = f"Camera {len(self.saved_cameras) + 1}"
                    # 리스트에 포즈 저장
                    self.saved_cameras.append(cam)
                    if len(self.saved_cameras) > 1:
                        # 첫 카메라가 아니라면 이전 카메라와의 전환 정보 기본값 추가
                        self.transition_times.append(2.0)   # 기본 이동 시간 2초
                        self.transition_frames.append(60)   # 기본 프레임 수 60
            # 저장된 카메라 목록 표시 및 개별 조작 버튼들
            for idx, cam in enumerate(list(self.saved_cameras)):
                imgui.bullet_text(f"{cam.name}")
                imgui.same_line()
                # ▲ 위로 이동 버튼 (첫 번째 항목에는 없음)
                if idx > 0:
                    if imgui_utils.button(f"Up##{idx}", width=viz.button_w * 0.5):
                        # 이전 항목과 교체
                        self.saved_cameras[idx-1], self.saved_cameras[idx] = self.saved_cameras[idx], self.saved_cameras[idx-1]
                        # 순서 변경 시 모든 전환 시간/프레임을 기본값으로 재설정
                        if len(self.saved_cameras) > 1:
                            self.transition_times = [2.0] * (len(self.saved_cameras) - 1)
                            self.transition_frames = [60] * (len(self.saved_cameras) - 1)
                        # 카메라 이름 다시 번호 매기기
                        break  # 목록이 변경되었으므로 루프 탈출
                else:
                    imgui.dummy(size=imgui.ImVec2(viz.button_w * 0.5, 0)) # 자리 맞춤용 빈 공간
                imgui.same_line()
                # ▼ 아래로 이동 버튼 (마지막 항목에는 없음)
                if idx < len(self.saved_cameras) - 1:
                    if imgui_utils.button(f"Down##{idx}", width=viz.button_w * 0.5):
                        # 다음 항목과 교체
                        self.saved_cameras[idx], self.saved_cameras[idx+1] = self.saved_cameras[idx+1], self.saved_cameras[idx]
                        if len(self.saved_cameras) > 1:
                            self.transition_times = [2.0] * (len(self.saved_cameras) - 1)
                            self.transition_frames = [60] * (len(self.saved_cameras) - 1)
                        break
                else:
                    imgui.dummy(size=imgui.ImVec2(viz.button_w * 0.5, 0))
                imgui.same_line()
                # 즉시 이동(GoTo) 버튼
                if imgui_utils.button(f"GoTo##{idx}", width=viz.button_w):
                    target_cam = cam
                    cam_widget = None
                    for w in viz.widgets:
                        if isinstance(w, camera.CamWidget):
                            cam_widget = w
                            break
                    if cam_widget is not None:
                        # CamWidget의 현재 상태를 해당 저장된 카메라로 설정
                        cam_widget.cam_pos = target_cam.cam_pos.clone() if hasattr(target_cam.cam_pos, "clone") else torch.tensor(target_cam.cam_pos, device=cam_widget.device)
                        cam_widget.up_vector = target_cam.up_vector.clone() if hasattr(target_cam.up_vector, "clone") else torch.tensor(target_cam.up_vector, device=cam_widget.device)
                        cam_widget.forward = target_cam.forward.clone() if hasattr(target_cam.forward, "clone") else torch.tensor(target_cam.forward, device=cam_widget.device)
                        cam_widget.pose.yaw = target_cam.yaw
                        cam_widget.pose.pitch = target_cam.pitch
                        cam_widget.radius = target_cam.radius
                        cam_widget.lookat_point = target_cam.lookat_point.clone() if hasattr(target_cam.lookat_point, "clone") else torch.tensor(target_cam.lookat_point, device=cam_widget.device)
                        cam_widget.current_control_mode = cam_widget.control_modes.index(target_cam.mode) if target_cam.mode in cam_widget.control_modes else cam_widget.current_control_mode
                        # 뷰어의 카메라 행렬 즉시 적용 (renderer에 전달될 값)
                        viz.args.cam_params = target_cam.cam_params
                        viz.args.yaw = target_cam.yaw
                        viz.args.pitch = target_cam.pitch
                        viz.args.fov = getattr(cam_widget, 'fov', 60)
                    # 외부 포즈 예측 기능 일시 정지 (있을 경우)
                    if hasattr(viz, 'auto_pose_enabled'):
                        viz.auto_pose_enabled = False
                imgui.same_line()
                # 삭제 버튼
                if imgui_utils.button(f"Delete##{idx}", width=viz.button_w):
                    # 선택한 카메라 포즈 삭제
                    self.saved_cameras.pop(idx)
                    if len(self.saved_cameras) == 0:
                        self.transition_times = []
                        self.transition_frames = []
                    else:
                        if idx == 0:
                            # 첫 번째 항목 삭제: 첫 번째 전환 구간 제거
                            self.transition_times.pop(0)
                            self.transition_frames.pop(0)
                        elif idx >= len(self.saved_cameras):
                            # 마지막 항목 삭제: 마지막 전환 구간 제거
                            self.transition_times.pop(-1)
                            self.transition_frames.pop(-1)
                        else:
                            # 중간 항목 삭제: 앞뒤 전환 구간 제거 후 새로운 구간 추가
                            self.transition_times.pop(idx)    # 삭제된 항목->다음 항목 구간 제거
                            self.transition_frames.pop(idx)
                            self.transition_times.pop(idx-1)  # 이전 항목->삭제된 항목 구간 제거
                            self.transition_frames.pop(idx-1)
                            # 삭제된 앞뒤를 연결하는 새 구간 기본값 추가
                            self.transition_times.insert(idx-1, 2.0)
                            self.transition_frames.insert(idx-1, 60)
                    break
            # 두 개 이상의 카메라 포즈가 있을 때만 재생 기능 UI 표시
            if len(self.saved_cameras) > 1:
                imgui.separator()
                imgui.text("Transitions between cameras:")
                # 각 연속하는 카메라 쌍의 전환 시간/프레임 입력 필드
                for i in range(len(self.saved_cameras) - 1):
                    imgui.push_id(f"seg{i}")
                    imgui.text(f"{self.saved_cameras[i].name} -> {self.saved_cameras[i+1].name}:")
                    imgui.same_line()
                    changed_time, new_time = imgui.input_float("sec", float(self.transition_times[i]), format="%.1f")
                    if changed_time:
                        self.transition_times[i] = max(new_time, 0.0)
                    imgui.same_line()
                    changed_frame, new_frames = imgui.input_int("frames", int(self.transition_frames[i]))
                    if changed_frame:
                        self.transition_frames[i] = max(new_frames, 1)
                    imgui.pop_id()
                # 재생 제어 버튼들
                if not self.playing:
                    # ▶ Play 버튼 (재생 시작)
                    if imgui_utils.button("Play", width=viz.button_w):
                        if len(self.saved_cameras) >= 2:
                            self.playing = True
                            self.paused = False
                            self.current_segment = 0
                            self._start_segment(0)                # 첫 번째 구간 초기화
                            if hasattr(viz, 'auto_pose_enabled'):
                                viz.auto_pose_enabled = False    # 자동 포즈 예측 비활성화
                            viz.playback_active = True            # 수동 카메라 조작 잠금
                else:
                    if not self.paused:
                        # 일시정지 ⏸
                        if imgui_utils.button("Pause", width=viz.button_w):
                            self.paused = True
                            self.pause_time = time.perf_counter()
                        imgui.same_line()
                        # 정지 ■ (재생 중단)
                        if imgui_utils.button("Stop", width=viz.button_w):
                            self.playing = False
                            self.paused = False
                            viz.playback_active = False
                            if hasattr(viz, 'auto_pose_enabled'):
                                viz.auto_pose_enabled = True
                    else:
                        # 일시정지 후 재시작 ▶
                        if imgui_utils.button("Resume", width=viz.button_w):
                            delta = time.perf_counter() - self.pause_time  # ★ 멈춰 있던 시간
                            self.segment_start_time += delta  # 재생 기준점을 뒤로 밀기
                            self.paused = False
                        imgui.same_line()
                        # 정지 ■
                        if imgui_utils.button("Stop", width=viz.button_w):
                            self.playing = False
                            self.paused = False
                            viz.playback_active = False
                            if hasattr(viz, 'auto_pose_enabled'):
                                viz.auto_pose_enabled = True
        if self.playing and not self.paused:
            # 모든 세그먼트를 이미 끝냈다면 종료
            if self.current_segment >= len(self.saved_cameras) - 1:
                self._stop_playback(viz)
                return

            # (1) 경과 시간 → 실제 진행률
            elapsed = time.perf_counter() - self.segment_start_time  # sec
            t_real = min(1.0, elapsed / self.seg_duration)  # 0‥1

            # (2) ‘frames’ 단위로 양자화(고정 FPS)
            frame_idx = int(t_real * self.current_frames)  # 0‥N
            if frame_idx == self.step_count:
                return  # 아직 새 프레임 시점이 아님 → 이전 행렬 유지
            self.step_count = frame_idx

            # (3) 보간 인수
            t = frame_idx / self.current_frames

            # ---------------- 회전·이동 보간 ----------------
            cos_ht = np.dot(self.q_start, self.q_end)
            if cos_ht > 0.9995:
                q = (1 - t) * self.q_start + t * self.q_end
            else:
                ht = math.acos(max(min(cos_ht, 1.0), -1.0))
                sth = math.sin(ht)
                ra = math.sin((1 - t) * ht) / sth
                rb = math.sin(t * ht) / sth
                q = self.q_start * ra + self.q_end * rb
            q /= np.linalg.norm(q)
            p = self.pos_start * (1 - t) + self.pos_end * t

            R = self._get_rotation_matrix_from_quaternion(q)
            extr = np.eye(4)
            extr[:3, :3] = R
            extr[:3, 3] = p
            viz.args.cam_params = torch.as_tensor(extr,
                                                  dtype=torch.float32,
                                                  device="cuda")
            # ★ CamWidget에도 같은 행렬을 복사
            if self.cam_widget:
                self.cam_widget.cam_params = viz.args.cam_params

            # (4) 세그먼트 종료?
            if frame_idx >= self.current_frames:
                # 마지막 프레임을 ‘종착 카메라’로 정확히 맞춤
                viz.args.cam_params = self.saved_cameras[self.current_segment + 1].cam_params

                # 다음 세그먼트로 넘어가기
                self.current_segment += 1
                if self.current_segment < len(self.saved_cameras) - 1:
                    self._start_segment(self.current_segment)  # 새 구간 초기화
                else:
                    self._stop_playback(viz)

            # ─────────────────── 보조 메서드 일부 발췌 ───────────────────

    def _start_segment(self, idx: int):
        """idx 카메라 → idx+1 카메라 보간 준비"""
        import time, numpy as np
        sc, ec = self.saved_cameras[idx], self.saved_cameras[idx + 1]
        R0, p0 = sc.cam_params[:3, :3].cpu().numpy(), sc.cam_params[:3, 3].cpu().numpy()
        R1, p1 = ec.cam_params[:3, :3].cpu().numpy(), ec.cam_params[:3, 3].cpu().numpy()
        q0 = self._get_quaternion_from_matrix(R0)
        q1 = self._get_quaternion_from_matrix(R1)
        if np.dot(q0, q1) < 0.0:
            q1 = -q1

        self.q_start, self.q_end = q0, q1
        self.pos_start, self.pos_end = p0, p1
        self.current_frames = max(1, int(self.transition_frames[idx]))
        self.seg_duration = max(1e-6, float(self.transition_times[idx]))
        self.segment_start_time = time.perf_counter()
        self.step_count = -1  # “아직 아무 프레임도 출력 안 함”
        # ── CamWidget 잠금 + 포인터 보관 ──
        self.cam_widget = next((w for w in self.viz.widgets
                                if isinstance(w, camera.CamWidget)), None)
        if self.cam_widget:
            self.cam_widget.locked_by_external = True

    def _stop_playback(self, viz):
        self.playing = self.paused = False
        if self.cam_widget:
            # ★ CamWidget 잠금 해제 전에 최종 행렬로 내부 필드 갱신
            final_mat = viz.args.cam_params.clone()
            self.cam_widget.set_external_camera_pose(final_mat.cpu())  # CamWidget에 내장된 헬퍼
            self.cam_widget.locked_by_external = False
        viz.playback_active = False
        if hasattr(viz, 'auto_pose_enabled'):
            viz.auto_pose_enabled = True
