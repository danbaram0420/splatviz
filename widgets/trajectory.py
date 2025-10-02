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
import os
try:
    import widgets.nimble_obj_fit as nimfit   # ✅ 모듈 전체 임포트
except Exception:
    import nimble_obj_fit as nimfit
from typing import Optional
from tkinter import Tk, filedialog
# -----------------------------
# 좌표계 유틸 (copy와 동일한 규약: XYZW)
# gaussian = R^T (bullet - p)
# -----------------------------
def _to_gaussian_pts(scene_quat_xyzw, scene_pos, pts_bullet):
    Rb = R.from_quat(np.asarray(scene_quat_xyzw, np.float64))  # (x,y,z,w)
    P  = np.asarray(pts_bullet, np.float64)
    t  = np.asarray(scene_pos,    np.float64)
    return Rb.inv().apply(P - t)

def _to_bullet_pts(scene_quat_xyzw, scene_pos, pts_gaussian):
    Rb = R.from_quat(np.asarray(scene_quat_xyzw, np.float64))  # (x,y,z,w)
    P  = np.asarray(pts_gaussian, np.float64)
    t  = np.asarray(scene_pos,     np.float64)
    return Rb.apply(P) + t

def _find_scene_obj_recursive(scene_path: str) -> Optional[str]:
    """scene_path(.ply) 주변에서 collision/scene/mesh 관련 .obj를 찾는다.
    - 현재 디렉터리 → 하위 재귀 → 상위 3단계까지 거슬러 올라가며 재귀 탐색
    """
    if not scene_path:
        return None
    start = os.path.abspath(os.path.dirname(scene_path))
    prefer = ("collision", "scene", "tsdf_fusion", "mesh", "vis")

    def scan_dir(root: str) -> Optional[str]:
        best, best_score = None, -1
        for r, _dirs, files in os.walk(root):
            for fn in files:
                if not fn.lower().endswith(".obj"):
                    continue
                score = 0
                low = fn.lower()
                for k in prefer:
                    if k in low:
                        score += 2
                # 루트와의 거리 짧을수록 가점
                rel = os.path.relpath(r, root)
                dist = 0 if rel == "." else len(rel.split(os.sep))
                score += max(0, 5 - dist)
                if score > best_score:
                    best_score = score
                    best = os.path.join(r, fn)
        return best

    # 1) 현재 폴더/하위
    cand = scan_dir(start)
    if cand:
        return cand
    # 2) 상위 폴더로 최대 3단계 상승하며 재귀 탐색
    up = start
    for _ in range(3):
        up = os.path.dirname(up)
        if not up or up == "/" or up == start:
            break
        cand = scan_dir(up)
        if cand:
            return cand
    return None

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

        # --- nimble 학습 관련 상태 ---
        self._training = False
        self.train_iters = 150  # 원하는 값으로 UI에서 바꿀 예정
        self.snap_every = 15  # n-iteration마다 스냅샷
        self._last_obj_path = None  # 방금 녹화한 OBJ 경로 기억
        self._last_init_quat = (0, 0, 0, 1)

        self.gt_rotx_deg = 0.0  # 사용자가 입력할 로테이션(deg, x축)
        self.gt_rotx_override = False  # 체크 시, 파일 메타/현 viz 회전 대신 이 값 사용

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

        # (TrajectoryWidget.__call__ 내부, Clear 버튼 아래쪽에 추가)
        if I.button("Load GT (.npy)", width=180, enabled=not self._recording_gt):
            try:
                # Tk 파일 다이얼로그(화면 깜빡임 방지)
                _tk = Tk();
                _tk.withdraw()
                npy_path = filedialog.askopenfilename(
                    title="Pick GT trajectory .npy",
                    filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")]
                )
                _tk.update();
                _tk.destroy()
                if npy_path:
                    self._load_gt_npy(npy_path)
            except Exception as e:
                print("[traj] load GT npy failed:", e)

        changed, val = imgui.input_float("GT rotx (deg)", float(self.gt_rotx_deg), 1.0, 5.0, "%.1f")
        if changed:
            self.gt_rotx_deg = float(val)

        clicked, checked = imgui.checkbox("Override rotation on load", bool(self.gt_rotx_override))
        if clicked:
            self.gt_rotx_override = bool(checked)

        if I.button("Choose & Load", width=180, enabled=not self._recording_gt):
            try:
                from tkinter import Tk, filedialog
                _tk = Tk();
                _tk.withdraw()
                npy_path = filedialog.askopenfilename(
                    title="Pick GT trajectory .npy",
                    filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")]
                )
                _tk.update();
                _tk.destroy()
                if npy_path:
                    self._load_gt_npy(npy_path,
                                      rotx_override_deg=(self.gt_rotx_deg if self.gt_rotx_override else None))
            except Exception as e:
                print("[traj] load GT npy failed:", e)

        imgui.separator()

        # 원본처럼: Load 위젯의 click-to-place를 그대로 사용
        if I.button("Create Trajectory", width=getattr(self.viz, "button_w", 480), enabled=not self._recording_gt):
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

            # ✨ 추가: 새 세션 번호 발급(증가)하고, 이번 아밍이 기다릴 세션을 저장
            self.viz.spawn_session_id += 1
            self._await_session = self.viz.spawn_session_id

            print("[traj] click the viewport to place the object.")

        # 상태 표시
        if self._recording_gt:
            imgui.same_line()
            imgui.text("Recording GT...")

        imgui.separator()
        imgui.text("Nimble parameter fit")
        ch, itv = imgui.input_int("iters", int(self.train_iters), 5, 25)
        if ch: self.train_iters = max(1, int(itv))
        ch, sev = imgui.input_int("snapshot every", int(self.snap_every), 1, 10)
        if ch: self.snap_every = max(1, int(sev))

        enabled_train = (not self._recording_gt) and (not self._training) and (len(self.viz.traj_gt) >= 4) and (
                    self._last_obj_path is not None)
        if I.button("Train (nimble)", width=200, enabled=enabled_train):
            import threading
            threading.Thread(target=self._train_nimble_worker, daemon=True).start()

        # --- 원본처럼: 프레임 업데이트에서만 진행 ---
        # 1) 클릭-소환 감지 → 녹화 시작  (세션 매칭 추가)
        if self._await_click_spawn:
            bid = getattr(self.viz, "last_spawned_bullet_id", None)
            sess = getattr(self.viz, "last_spawn_session_id", -2)
            aw = getattr(self, "_await_session", -1)
            if bid is not None and sess == aw:
                # 이번 Create Trajectory에서 소환된 것이 맞다 → 녹화 시작
                self._await_click_spawn = False
                self.viz.last_spawned_bullet_id = None  # 다음 감지를 위해 비움
                self._start_gt_recording(int(bid))

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

        # --- [추가] 방금 소환한 동적체의 OBJ 경로 기억 ---
        self._last_obj_path = None
        try:
            dyn_map = getattr(self.viz, "dynamic_objects", {})
            for ply_path, inf in dyn_map.items():
                if inf.get("id") == self._tracked_bid:
                    self._last_obj_path = inf.get("obj_path", None)  # load_ply가 넣어둔 필드 이름
                    break
        except Exception:
            pass

        self._last_init_quat = tuple(orn)  # nimble urdf rpy 변환에 사용

        # 공유 버퍼/상태
        self.viz.traj_gt.clear()
        self._start_time = time.time()
        self._frame_count = 0
        self._recording_gt = True
        self.viz.traj_recording = True
        self.viz.traj_viz_on = True
        # downsample/length는 UI 반영 값이 이미 self.viz에 들어가 있음

        try:
            self.viz.traj_dt = float(p.getPhysicsEngineParameters().get("fixedTimeStep", 0.004))
        except Exception:
            self.viz.traj_dt = float(self.viz.traj_length) / max(1, (len(self.viz.traj_gt) - 1))

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
        print(f"[traj] sample #{len(store_list)}  pt={pt_g.tolist()[:3]}")

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

    def _load_gt_npy(self, path: str, rotx_override_deg: float | None = None):
        import numpy as np
        from scipy.spatial.transform import Rotation as R

        arr = np.load(path, allow_pickle=True)
        meta = {}

        # --- 다양한 저장 포맷 허용: dict/npz/ndarray ---
        if isinstance(arr, np.lib.npyio.NpzFile):
            if "traj" in arr:
                data = arr["traj"]
            elif "gt_traj" in arr:
                data = arr["gt_traj"]
            elif "pos" in arr:
                data = arr["pos"]
            else:
                data = list(arr.values())[0]
            for k in ("dt", "h", "stride", "t0", "t1",
                      "scene_origin_pos", "scene_origin_quat_xyzw", "scene_rotx_deg", "origin"):
                if k in arr: meta[k] = arr[k]
        elif isinstance(arr, np.ndarray):
            data = arr
        else:
            d = arr.item()  # dict
            data = d.get("traj", d.get("gt_traj", d.get("pos", None)))
            if data is None:
                raise RuntimeError("Unsupported dict fields in npy")
            for k in ("dt", "h", "stride", "t0", "t1",
                      "scene_origin_pos", "scene_origin_quat_xyzw", "scene_rotx_deg", "origin"):
                if k in d: meta[k] = d[k]

        data = np.asarray(data, np.float64)

        # (T,3) 또는 (T,N,3) → 우선 0번만 사용 (멀티 오버레이는 이후 확장)
        if data.ndim == 3:
            data = data[:, 0, :]
        assert data.ndim == 2 and data.shape[1] == 3, f"GT traj must be (T,3) or (T,N,3), got {data.shape}"

        # --- 씬 포즈 결정: meta → viz → override 순 ---
        # meta가 numpy 배열일 수 있으므로 np.asarray 처리
        scene_q_viz = np.asarray(getattr(self.viz, "scene_origin_quat", [0, 0, 0, 1]), np.float64)
        scene_p_viz = np.asarray(getattr(self.viz, "scene_origin_pos", [0, 0, 0]), np.float64)

        scene_q = scene_q_viz
        scene_p = scene_p_viz

        if "scene_origin_quat_xyzw" in meta:
            q = np.asarray(meta["scene_origin_quat_xyzw"], np.float64).reshape(4, )
            scene_q = q
        if "scene_origin_pos" in meta:
            p = np.asarray(meta["scene_origin_pos"], np.float64).reshape(3, )
            scene_p = p

        # 사용자가 오버라이드 체크했으면: viz/meta 무시하고 x-축 회전deg로 세팅
        if rotx_override_deg is not None:
            Rx = R.from_euler("x", float(rotx_override_deg), degrees=True)
            scene_q = Rx.as_quat()  # (x,y,z,w)

        # Bullet → Gaussian 변환
        Rb = R.from_quat(scene_q)  # scene(Gaussian→Bullet)
        pts_g = Rb.inv().apply(data - scene_p)

        # --- (선택) COM/BASE 기준 보정: origin 메타가 'com'이면 여기서 처리 가능 ---
        # * 권장: GT는 항상 'base'로 저장. 'com'일 경우 per-frame orn이 없으면 정확 복원 불가.
        # if meta.get("origin","base") == "com":
        #     print("[traj] GT is COM-based; per-frame orientation required to convert to base correctly.")

        # viz 상태 갱신
        self.viz.traj_gt = [tuple(p) for p in pts_g.tolist()]
        self.viz.traj_viz_on = True
        if "dt" in meta:
            self.viz.traj_dt = float(meta["dt"])
        elif "h" in meta:
            self.viz.traj_dt = float(meta["h"])

        print(f"[traj] loaded GT: {len(self.viz.traj_gt)} points from '{path}' "
              f"(rotx_override={rotx_override_deg}, meta_rotx={float(meta.get('scene_rotx_deg', np.nan)) if 'scene_rotx_deg' in meta else None})")

    def _train_nimble_worker(self):
        if self._training:
            return
        self._training = True
        try:
            # 안전 체크
            if self._last_obj_path is None or len(self.viz.traj_gt) < 4:
                print("[nimble] need GT and obj_path")
                return

            dt = float(getattr(self.viz, "traj_dt", 0.004))

            traj_g = np.asarray(self.viz.traj_gt, dtype=np.float32)

            # 1) Gaussian→Bullet
            scene_q = np.asarray(getattr(self.viz, "scene_origin_quat", [0, 0, 0, 1]), np.float64)
            scene_p = np.asarray(getattr(self.viz, "scene_origin_pos", [0, 0, 0]), np.float64)
            traj_b = _to_bullet_pts(scene_q, scene_p, np.asarray(self.viz.traj_gt, np.float32))

            # 2) 씬 OBJ 후보 자동 탐색 (scene_path 옆)
            scene_obj = _find_scene_obj_recursive(getattr(self.viz, "scene_path", ""))
            print("[nimble] scene candidate:", scene_obj)
            est, pred_b, snaps_b = nimfit.fit_params_from_gt(
                gt_xyz=traj_b, dt=float(self.viz.traj_dt),
                obj_path=self._last_obj_path,
                init_pose_xyz=tuple(traj_b[0]),
                init_quat_xyzw=tuple(self._last_init_quat),
                scene_obj=scene_obj,
                scene_pos_xyz=tuple(scene_p.tolist()),
                scene_quat_xyzw=tuple(scene_q.tolist()),
                use_plane=(scene_obj is None),
                iters=int(self.train_iters),
                record_every=int(self.snap_every),
                seed=1234,
                init_guess=None,
            )

            # 예측/스냅샷을 Gaussian으로 되돌려 viz에 반영
            pred_g = _to_gaussian_pts(scene_q, scene_p, np.asarray(pred_b, np.float64))
            self.viz.traj_pred = [tuple(p) for p in pred_g]
            self.viz.traj_viz_on = True

            snaps_g = []
            for s in snaps_b:
                s_g = _to_gaussian_pts(scene_q, scene_p, np.asarray(s, np.float64))
                snaps_g.append((s_g.astype(np.float32), [1.0, 0.5, 0.0, 1.0]))  # 색상은 뷰어가 무시할 수도 있음
            # 뷰어가 (points, color) 튜플 리스트를 읽는 구조라면 그대로, 아니라면 points만 넣어도 됨
            self.viz.traj_snapshots = snaps_g if hasattr(self.viz, "traj_snapshots") else []

            # 학습 결과 저장(뷰어 패널에서 확인 가능)
            self.viz.traj_learned_params = est
            print("[nimble] est:", est)

        except Exception as e:
            print("[nimble] train error:", e)
        finally:
            self._training = False