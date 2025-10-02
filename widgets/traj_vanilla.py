# widgets/trajectory.py
# -*- coding: utf-8 -*-
"""
Trajectory recording & visualization ONLY (copy와 동일한 흐름)
- 클릭 기반 소환 → spawn 감지 → Bullet에서 프레임 단위 샘플 → Gaussian 좌표로 변환하여 viz.traj_gt에 저장
- 녹화 종료 시 Bullet 바디와 Gaussian(.ply) 모두 정리
- 오버레이(시각화)는 splatviz가 viz.traj_gt를 읽어 화면에 그림
"""
import time
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from imgui_bundle import imgui
from splatviz_utils.gui_utils import imgui_utils as I  # 프로젝트 표준 버튼/헬퍼만 사용

# -----------------------------
# 좌표계 유틸 (copy와 동일한 규약: XYZW)
# gaussian = R^T (bullet - p)
# -----------------------------
def _to_gaussian_pts(scene_quat_xyzw, scene_pos, pts_bullet):
    Rb = R.from_quat(np.asarray(scene_quat_xyzw, np.float64))  # (x,y,z,w)
    P  = np.asarray(pts_bullet, np.float64)
    t  = np.asarray(scene_pos,    np.float64)
    return Rb.inv().apply(P - t)

class TrajectoryWidget:
    def __init__(self, viz):
        self.viz = viz
        self.name = "Trajectory"

        # UI
        self.downsample  = 1      # 프레임 단위 다운샘플(오버레이 표시 간격)
        self.length_sec  = 5.0    # 녹화 길이(초)

        # 상태
        self._await_click_spawn = False
        self._tracked_bid  = None
        self._start_time   = 0.0
        self._frame_count  = 0
        self._recording_gt = False

        # splatviz가 기대하는 필드들만 (없을 때에만) 초기화
        defaults = [
            ("traj_gt", []),
            ("traj_pred", []),
            ("traj_viz_on", True),
            ("traj_downsample", 1),
            ("traj_length", float(self.length_sec)),
            ("traj_recording", False),
            # scene_origin_* 는 splatviz가 이미 설정하므로 여기서 새로 지정/덮어쓰지 않음
        ]
        for k, v in defaults:
            if not hasattr(self.viz, k):
                setattr(self.viz, k, v)

    @I.scoped_by_object_id
    def __call__(self, expanded):
        if not expanded:
            return

        # --- 최소 UI (copy 풍) ---
        imgui.text("Record seconds:")
        ch, val = imgui.input_float("##traj_secs", float(self.length_sec), 0.1, 1.0, "%.2f")
        if ch:
            self.length_sec = max(0.1, float(val))
            self.viz.traj_length = float(self.length_sec)

        imgui.text("Overlay downsample (frames):")
        ch, v = imgui.input_int("##traj_down", int(self.downsample), 1, 5)
        if ch:
            self.downsample = max(1, int(v))
            self.viz.traj_downsample = self.downsample

        imgui.separator()
        _, self.viz.traj_viz_on = imgui.checkbox("Show Trajectory Overlay", bool(self.viz.traj_viz_on))
        imgui.same_line()
        if I.button("Clear", width=120):
            self.viz.traj_gt = []
            self.viz.traj_pred = []
            if hasattr(self.viz, "traj_snapshots"):
                self.viz.traj_snapshots = []

        imgui.separator()

        # 원본처럼: Load 위젯의 click-to-place를 그대로 사용
        if I.button("Create Trajectory", width=getattr(self.viz, "button_w", 240), enabled=not self._recording_gt):
            # Load 위젯이 준비한 (PLY, OBJ) 페어를 등록 (없으면 None을 두고도 arming은 가능)
            loadw = self.viz.widgets[0] if getattr(self.viz, "widgets", None) else None
            prep  = getattr(loadw, "prepare_object_files_for_insertion", None) if loadw else None
            if callable(prep):
                try:
                    file_path, obj_path = prep()
                    if file_path and obj_path:
                        self.viz.pending_spawn_files = (file_path, obj_path)
                except Exception:
                    pass
            # 클릭-소환 arming: 실제 스폰은 splatviz가 하고, 여기서는 id 감지만 한다
            self.viz.awaiting_spawn_click = True
            self._await_click_spawn = True
            self.viz.traj_viz_on = True
            self.viz.traj_gt.clear(); self.viz.traj_pred.clear()
            self._tracked_bid = None
            self._start_time = 0.0
            self._frame_count = 0
            print("[traj] click the viewport to place the object.")

        # 상태 표시
        if self._recording_gt:
            imgui.same_line()
            imgui.text("Recording GT...")

        # --- 원본처럼: 프레임 업데이트에서만 진행 ---
        # 1) 클릭-소환 감지 → 녹화 시작
        if self._await_click_spawn and getattr(self.viz, "last_spawned_bullet_id", None) is not None:
            self._await_click_spawn = False
            bid = int(self.viz.last_spawned_bullet_id)
            self.viz.last_spawned_bullet_id = None  # 다음 감지를 위해 비움
            self._start_gt_recording(bid)

        # 2) 녹화 중이면 프레임마다 샘플링 (다운샘플 반영)
        if self._recording_gt and self._tracked_bid is not None:
            elapsed = time.time() - self._start_time
            if elapsed >= float(self.viz.traj_length):
                self._stop_gt_recording()
            else:
                self._maybe_sample_current(self._tracked_bid, self.viz.traj_gt)

    # ---------- GT 녹화 ----------
    def _start_gt_recording(self, bid: int):
        """스폰된 바디(bid)를 프레임 단위로 샘플링하여 Gaussian 좌표의 GT 궤적을 남긴다."""
        self._tracked_bid = int(bid)

        # 초기 상태 저장(원본과 동일 필드; 이후 nimble 통합 시 사용 가능)
        try:
            pos, orn = p.getBasePositionAndOrientation(self._tracked_bid)
            lin, ang = p.getBaseVelocity(self._tracked_bid)
        except Exception:
            pos, orn, lin, ang = (0,0,0), (0,0,0,1), (0,0,0), (0,0,0)
        self.viz.traj_init_state = dict(pos=pos, orn=orn, lin=lin, ang=ang)

        # 공유 버퍼/상태
        self.viz.traj_gt.clear()
        self._start_time = time.time()
        self._frame_count = 0
        self._recording_gt = True
        self.viz.traj_recording = True
        self.viz.traj_viz_on = True
        # downsample/length는 UI 반영 값이 이미 self.viz에 들어가 있음

    def _maybe_sample_current(self, bid: int, store_list: list):
        """현재 Bullet 바디 위치를 sample하여 Gaussian 좌표로 누적."""
        self._frame_count += 1
        if (self._frame_count % int(max(1, self.viz.traj_downsample))) != 0:
            return

        try:
            pos_b, _ = p.getBasePositionAndOrientation(bid)
        except Exception:
            self._stop_gt_recording()
            return

        # scene_origin_* 는 splatviz가 유지하므로 그대로 사용 (절대 덮어쓰지 않음)
        scene_p = getattr(self.viz, "scene_origin_pos",  [0,0,0])
        scene_q = getattr(self.viz, "scene_origin_quat", [0,0,0,1])  # XYZW
        pt_g = _to_gaussian_pts(scene_q, scene_p, np.asarray([pos_b], np.float64))[0]
        store_list.append(pt_g.tolist())

    def _stop_gt_recording(self):
        self._recording_gt = False
        self.viz.traj_recording = False
        # 원본처럼: 녹화가 끝나면 동적 바디와 대응 PLY를 정리
        try:
            if self._tracked_bid is not None and hasattr(self.viz, "remove_dynamic_object_by_bid"):
                self.viz.remove_dynamic_object_by_bid(self._tracked_bid, remove_ply=True)
        except Exception as e:
            print("[trajectory] remove GT object error:", e)
        self._tracked_bid = None
