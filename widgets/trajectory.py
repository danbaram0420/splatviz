# widgets/trajectory.py
from imgui_bundle import imgui
import time
import threading
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R  # ← 추가
from widgets.load_ply import create_physics_object_from_mesh
import math

from splatviz_utils.gui_utils import imgui_utils  # 프로젝트에 이미 있는 헬퍼
# from .easy_imgui import label  # label 쓰신다면 주석 해제

# 선택 의존성(없으면 폴백)
try:
    import cma
except Exception:
    cma = None
try:
    from dtaidistance import dtw
except Exception:
    dtw = None

# ====== Single-body quick-fit helpers (비블로킹 사용) ======

_BULLET_DT = 1.0 / 240.0

def _downsample_nd(traj_np: np.ndarray, stride: int) -> np.ndarray:
    stride = max(1, int(stride))
    return traj_np[::stride].copy()

def _traj_loss(gt_xyz: np.ndarray, sim_xyz: np.ndarray, use_dtw: bool) -> float:
    """두 궤적(각 N×3)을 비교. DTW가 있으면 DTW, 없으면 L2."""
    if gt_xyz.shape[0] < 2 or sim_xyz.shape[0] < 2:
        return 1e9
    if use_dtw and (dtw is not None):
        a = np.ascontiguousarray(gt_xyz[:, [0,2]], dtype=np.double)  # X,Z 우선
        b = np.ascontiguousarray(sim_xyz[:,[0,2]], dtype=np.double)
        # 1D DTW 2축 합산(간단)
        L = float(dtw.distance_fast(a[:,0], b[:,0], use_pruning=True)) \
          + float(dtw.distance_fast(a[:,1], b[:,1], use_pruning=True))
        # Y는 약하게 가중
        L += 0.3 * float(dtw.distance_fast(np.ascontiguousarray(gt_xyz[:,1], dtype=np.double),
                                           np.ascontiguousarray(sim_xyz[:,1], dtype=np.double),
                                           use_pruning=True))
        return L
    # 폴백: 평균 제곱 오차
    N = min(len(gt_xyz), len(sim_xyz))
    return float(np.mean((gt_xyz[:N] - sim_xyz[:N])**2))

def _simulate_single_body(params, steps, init_height, init_v0_xyz, mode=p.DIRECT):
    """
    PyBullet DIRECT로 단일 바디를 빠르게 시뮬레이션.
    params = (restitution, lateralFriction, mass)
    반환: (steps,3) 위치 배열 (Bullet world)
    """
    restitution, mu, mass = float(params[0]), float(params[1]), float(params[2])
    mass = float(np.clip(mass, 0.1, 20.0))
    mu   = float(np.clip(mu,   0.0, 1.0))
    rest = float(np.clip(restitution, 0.0, 1.0))

    cid = p.connect(mode)
    if cid < 0:
        # 드물게 실패 시 재시도
        time.sleep(0.02)
        cid = p.connect(mode)
    try:
        p.setGravity(0,0,-9.81, physicsClientId=cid)
        p.setTimeStep(_BULLET_DT, physicsClientId=cid)
        plane = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=cid)
        p.createMultiBody(0.0, plane, physicsClientId=cid)

        # 충돌형상: 구(0.1m) — 빠르고 안정적
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.10, physicsClientId=cid)
        bid = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                                basePosition=[0.0, 0.0, float(init_height)],
                                physicsClientId=cid)
        p.changeDynamics(bid, -1, restitution=rest, lateralFriction=mu, physicsClientId=cid)
        p.resetBaseVelocity(bid, linearVelocity=list(map(float, init_v0_xyz)), physicsClientId=cid)

        traj = np.zeros((steps,3), dtype=np.float64)
        for i in range(steps):
            p.stepSimulation(physicsClientId=cid)
            pos,_ = p.getBasePositionAndOrientation(bid, physicsClientId=cid)
            traj[i] = pos
        return traj
    finally:
        p.disconnect(physicsClientId=cid)

def _estimate_v0_from_traj(gt_xyz: np.ndarray, length_sec: float) -> np.ndarray:
    """GT(렌더 좌표)에서 첫 두 샘플로 초기 속도 근사."""
    if len(gt_xyz) >= 2 and length_sec > 0:
        dt_est = length_sec / max(1, (len(gt_xyz)-1))
        v = (gt_xyz[1] - gt_xyz[0]) / max(1e-6, dt_est)
        return v
    return np.zeros(3, dtype=np.float64)

def _fit_single_body_quick(gt_gauss_pts: list, length_sec: float, ds: int,
                           popsize: int = 10, maxiter: int = 20, cpu_workers: int = 1):
    """
    빠른 단일바디 피팅(반발계수,마찰,질량).
    - GT는 '렌더 좌표계'로 들어옴. 절대 위치 보정을 위해 첫 점을 원점으로 정렬 후 비교.
    - cma가 있으면 CMA-ES, 없으면 소규모 랜덤 탐색 → 자잘한 폴리싱.
    반환: dict {'restitution','lateralFriction','mass'}
    """
    gt = np.asarray(gt_gauss_pts, dtype=np.float64)
    if len(gt) < 3:
        return {'restitution':0.3, 'lateralFriction':0.4, 'mass':1.0}

    # 원점 정렬(절대 위치 영향 제거) + 다운샘플
    gt_rel = gt - gt[0]
    gt_rel = _downsample_nd(gt_rel, max(1, int(ds)))

    # 시뮬 길이/스텝 설정 + v0 근사
    steps  = max(8, int(length_sec / _BULLET_DT))
    v0_est = _estimate_v0_from_traj(gt, length_sec)

    # 목적함수 (렌더좌표 vs Bullet 좌표 비교는 '상대 궤적'으로 흡수)
    def obj(x):
        # x = [rest, mu, mass]
        sim = _simulate_single_body(x, steps, init_height=max(0.05, gt[0,2]),
                                    init_v0_xyz=v0_est, mode=p.DIRECT)
        sim_rel = sim - sim[0]
        sim_rel = _downsample_nd(sim_rel, max(1, int(ds)))
        return _traj_loss(gt_rel, sim_rel, use_dtw=True)

    # 초기치/경계
    x_best = np.array([0.4, 0.4, 1.0], dtype=np.float64)
    lo = np.array([0.0, 0.0, 0.1], dtype=np.float64)
    hi = np.array([1.0, 1.0, 20.0], dtype=np.float64)

    if cma is not None:
        # 경량 CMA (재시작 없음, 소인구, 제한 세대)
        opts = {
            'popsize': popsize,
            'maxiter': maxiter,
            'verb_log':0, 'verb_disp':0,
            'bounds': [lo.tolist(), hi.tolist()],
            'CMA_stds': [0.2, 0.2, 0.5],
            'tolx': 1e-4, 'tolfun': 1e-5,
        }
        es = cma.CMAEvolutionStrategy(x_best, 0.2, opts)
        gen = 0
        while not es.stop():
            gen += 1
            X = es.ask()
            # 렌더링 끊김 방지를 위해 '직렬' 평가(기본값). 필요 시 cpu_workers>1로 풀 수 있음.
            F = [obj(xx) for xx in X] if cpu_workers <= 1 else list(map(obj, X))
            es.tell(X, F)
            # 아주 잘 맞으면 조기 종료
            if es.result.fbest < 1e-3:
                break
        x_best = es.result.xbest
    else:
        # 폴백: 작은 격자 + 주변 미세 탐색
        grid_r = np.linspace(0.1, 0.9, 5)
        grid_m = np.linspace(0.05, 0.8, 6)
        grid_M = np.linspace(0.5, 5.0, 6)
        bestL = 1e9
        for r in grid_r:
            for m in grid_m:
                for M in grid_M:
                    L = obj((r,m,M))
                    if L < bestL: bestL, x_best = L, (r,m,M)
        for _ in range(20):
            cand = np.array(x_best) + np.random.normal(0, [0.05,0.05,0.2])
            cand = np.minimum(hi, np.maximum(lo, cand))
            Lc = obj(cand)
            if Lc < obj(x_best):
                x_best = cand

    rest, mu, mass = float(x_best[0]), float(x_best[1]), float(x_best[2])
    return {'restitution': rest, 'lateralFriction': mu, 'mass': mass}

class TrajectoryWidget:
    """
    - Create Trajectory: 클릭-투-플레이스로 오브젝트 소환 후 Length초 동안 GT 궤적 기록 (downsample 반영)
    - Train: 백그라운드에서 학습(미제공시 더미), 학습완료 후 같은 위치에서 예측 궤적을 Length초 기록
    - Visualize Trajectory: 토글 (GT=파랑, Pred=빨강) — 실제 그리기는 splatviz.draw_frame 오버레이에서 수행
    """
    def __init__(self, viz):
        self.viz = viz
        self.name = "Trajectory"

        # UI 상태
        self.downsample = 1
        self.length_sec = 5.0
        self.viz_toggle = False

        # 내부 상태
        self._await_click_spawn = False
        self._tracked_bid = None
        self._start_time = 0.0
        self._frame_count = 0
        self._recording_gt = False
        self._recording_pred = False

        # 학습 상태
        self._training_thread = None
        self._training_in_progress = False

        self._tracked_com = None
        self._tracked_quatI = None

        self.track_com_only = True

        self._tracked_ply_path = None
        self._tracked_obj_path = None
        self._pred_bid = None  # 예측 시 임시 바디

        self.fit_cpu_workers = 1  # 렌더 끊김 방지: 직렬 평가(필요하면 2~4로 올려도 됨)
        self.fast_fit_popsize = 10  # CMA 소인구
        self.fast_fit_maxiter = 20  # CMA 세대 수

    def _start_gt_recording(self, bid: int):
        """GT 녹화 시작: 현재 길이/다운샘플 파라미터로 초기화"""
        self._tracked_bid = bid

        # ★ 렌더와 동일 좌표계를 위해 관성 오프셋(com/quat_I) 확보
        info = None
        for _path, inf in self.viz.dynamic_objects.items():
            if inf.get("id") == bid:
                info = inf
                break
        if info is not None:
            self._tracked_com = np.asarray(info["com"], dtype=np.float64)
            self._tracked_quatI = np.asarray(info["quat_I"], dtype=np.float64)  # XYZW!
        else:
            # 안전 폴백(Quat는 XYZW 순서): [0,0,0,1]
            self._tracked_com = np.zeros(3, dtype=np.float64)
            self._tracked_quatI = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        # info 찾아낸 바로 아래에 추가
        self._tracked_ply_path = None
        self._tracked_obj_path = None
        for _path, inf in self.viz.dynamic_objects.items():
            if inf.get("id") == bid:
                self._tracked_ply_path = _path  # ← ply 경로
                self._tracked_obj_path = inf.get("obj_path")  # ← obj 경로(있으면)
                break

        # 초기 상태 저장 (예측 시작 시 reset용)
        pos, orn = p.getBasePositionAndOrientation(bid)
        linvel, angvel = p.getBaseVelocity(bid)
        # ★ GT 물성(현재 바디 dynamics) 캐싱: mass / lateralFriction / restitution
        try:
            dinfo = p.getDynamicsInfo(bid, -1)
            gt_params = {
                "mass": float(dinfo[0]) if len(dinfo) > 0 else None,
                "lateralFriction": float(dinfo[1]) if len(dinfo) > 1 else None,
                "restitution": float(dinfo[5]) if len(dinfo) > 5 else None,
            }
        except Exception:
            gt_params = {}
        self.viz.traj_gt_params = gt_params
        self.viz.traj_init_state = dict(pos=pos, orn=orn, lin=linvel, ang=angvel)

        self.viz.traj_gt.clear()
        self.viz.traj_pred.clear()
        self._start_time = time.time()
        self._frame_count = 0
        self._recording_gt = True
        self.viz.traj_recording = True
        self.viz.traj_downsample = int(max(1, self.downsample))
        self.viz.traj_length = float(max(0.1, self.length_sec))

    def _maybe_sample_current(self, bid: int, store_list: list):
        """다운샘플 주기에 맞춰 현재 '렌더 기준(Gaussian world)' 위치 1개를 기록"""
        self._frame_count += 1
        if (self._frame_count % self.viz.traj_downsample) != 0:
            return

        pos_t, quat_t = p.getBasePositionAndOrientation(bid)  # pos_t: COM (bullet world), quat_t: XYZW

        # 렌더(gaussian) 좌표계 정보
        scene_pos = getattr(self.viz, "scene_origin_pos", np.zeros(3, dtype=np.float64))
        scene_quat = getattr(self.viz, "scene_origin_quat", [0.0, 0.0, 0.0, 1.0])
        Rb_inv = R.from_quat(scene_quat).inv()

        if self.track_com_only:
            # ✅ 회전 완전 무시: COM만 기록 (가장 안정적)
            p_gauss = Rb_inv.apply(np.asarray(pos_t, dtype=np.float64) - scene_pos)
        else:
            # (이전 방식) 시각 모델 원점(visual origin) 기준 — 회전시 원그림/들썩임이 생길 수 있음
            if self._tracked_com is not None and self._tracked_quatI is not None:
                R_bt = R.from_quat(quat_t) * R.from_quat(self._tracked_quatI).inv()
                p_vis_bullet = np.asarray(pos_t, dtype=np.float64) - R_bt.apply(self._tracked_com)
            else:
                p_vis_bullet = np.asarray(pos_t, dtype=np.float64)
            p_gauss = Rb_inv.apply(p_vis_bullet - scene_pos)

        store_list.append(p_gauss.tolist())

    def _stop_gt_recording(self):
        self._recording_gt = False
        self.viz.traj_recording = False
        # ★ 녹화 끝: 바디/가우시안 제거
        if self._tracked_bid is not None:
            try:
                self.viz.remove_dynamic_object_by_bid(self._tracked_bid, remove_ply=True)
            except Exception as e:
                print("[trajectory] remove GT object error:", e)
        self._tracked_bid = None

    def _start_pred_recording(self):
        """학습 완료 후 예측 궤적 기록 시작: 임시 바디 생성 → 물성 적용 → 기록 후 제거"""
        # GT를 제거한 상태여야 하고, obj_path 정보가 있어야 한다.
        if self._tracked_obj_path is None:
            # obj_path가 없으면 기존 바디를 재활용할 수 없으므로, 예측 스킵
            # (원하면 여기서 기본 구체를 만들어도 됨)
            return

        s = self.viz.traj_init_state or {}
        init_pos = s.get("pos", [0, 0, 0])
        init_orn = self.viz.scene_origin_quat  # 물체 자체는 scene 전역 회전으로 고정 생성
        try:
            # 1) 임시 바디 생성 (Gaussian 등록 안 함 → 렌더에 영향 없음)
            bid, com, quat_I = create_physics_object_from_mesh(self._tracked_obj_path,
                                                               self.viz.scene_origin_quat,
                                                               np.asarray(init_pos, dtype=np.float64))
            self._pred_bid = bid
            # 2) 초기 포즈/속도 맞추기
            p.resetBasePositionAndOrientation(bid, init_pos, s.get("orn", [0, 0, 0, 1]))
            p.resetBaseVelocity(bid, s.get("lin", [0, 0, 0]), s.get("ang", [0, 0, 0]))
            # 3) 학습 결과 물성 적용
            params = self.viz.traj_learned_params or {}
            kwargs = {}
            if "mass" in params:
                kwargs["mass"] = float(params["mass"])
            if "lateralFriction" in params:
                kwargs["lateralFriction"] = float(params["lateralFriction"])
            if "restitution" in params:  # ★ 추가: 반발계수 적용
                kwargs["restitution"] = float(params["restitution"])
            if kwargs:
                p.changeDynamics(bid, -1, **kwargs)
            # 4) 예측 기록 초기화
            self.viz.traj_pred.clear()
            self._start_time = time.time()
            self._frame_count = 0
            self._recording_pred = True
        except Exception as e:
            print("[trajectory] pred spawn error:", e)
            self._pred_bid = None
            self._recording_pred = False

    def _stop_pred_recording(self):
        self._recording_pred = False
        # ★ 기록 끝: 임시 바디 제거 (Gaussian 등록 안 했음)
        if self._pred_bid is not None:
            try:
                p.removeBody(self._pred_bid)
            except Exception:
                pass
        self._pred_bid = None

    # -------------------- UI / 프레임 루프 --------------------

    def __call__(self, expanded):
        # 패널 헤더
        if imgui.collapsing_header("Trajectory", imgui.TreeNodeFlags_.default_open):
            # Downsample / Length
            imgui.text("Downsample (frames):")
            changed, val = imgui.input_int("##traj_down", self.downsample)
            if changed:
                self.downsample = max(1, int(val))

            imgui.text("Length (sec):")
            changed, valf = imgui.input_float("##traj_len", self.length_sec, 0.1, 1.0, "%.2f")
            if changed:
                self.length_sec = max(0.1, float(valf))
            changed, tval = imgui.checkbox("Track COM (ignore rotation)", self.track_com_only)
            if changed:
                self.track_com_only = bool(tval)

            # Create Trajectory
            if imgui_utils.button("Create Trajectory", width=self.viz.button_w):
                # 1) 먼저 이전 스폰 잔여값을 비워 즉시 시작 방지
                self.viz.last_spawned_bullet_id = None

                # 2) 클릭-투-플레이스 준비 (load_ply 헬퍼 재사용)
                loadw = self.viz.widgets[0]  # 보통 Load/Insert 위젯이 0번
                prep = getattr(loadw, "prepare_object_files_for_insertion", None)
                if prep is not None:
                    file_path, obj_path = prep()
                    if file_path and obj_path:
                        self.viz.pending_spawn_files = (file_path, obj_path)
                        self.viz.awaiting_spawn_click = True
                else:
                    # 최소: 클릭-소환 모드만 켠다
                    self.viz.awaiting_spawn_click = True

                # 3) "클릭으로 실제 스폰"을 기다렸다가 녹화 시작
                self._await_click_spawn = True
                self._tracked_bid = None

            imgui.same_line()
            # Train
            train_disabled = not (len(self.viz.traj_gt) > 0 and not self._recording_gt and not self._training_in_progress)
            # --- Train 버튼 ---
            if imgui.button("Train") and (not getattr(self, "_training_in_progress", False)):
                self._training_in_progress = True
                self.viz.traj_training_done = False

                # 백그라운드 워커(렌더 끊김 방지)
                def _worker():
                    try:
                        gt_pts = list(self.viz.traj_gt)  # [(x,y,z),... in Gaussian coords]
                        if len(gt_pts) < 3:
                            raise RuntimeError("Not enough GT points. Record longer trajectory.")

                        ds = int(max(1, self.downsample))
                        Lsec = float(max(0.1, self.length_sec))

                        # ===== 1) 학습 =====
                        learned = _fit_single_body_quick(
                            gt_gauss_pts=gt_pts, length_sec=Lsec, ds=ds,
                            popsize=int(getattr(self, "fast_fit_popsize", 10)),
                            maxiter=int(getattr(self, "fast_fit_maxiter", 20)),
                            cpu_workers=int(getattr(self, "fit_cpu_workers", 1)),
                        )
                        self.viz.traj_learned_params = {
                            'lateralFriction': learned.get('lateralFriction', 0.4),
                            'restitution': learned.get('restitution', 0.3),
                            'mass': learned.get('mass', 1.0),
                        }

                        # ===== 2) 로스 계산 & 프린트 =====
                        import numpy as _np
                        gt = _np.asarray(gt_pts, dtype=_np.float64)
                        gt_rel = gt - gt[0]
                        gt_rel = _downsample_nd(gt_rel, ds)

                        steps = max(8, int(Lsec / _BULLET_DT))
                        v0_est = _estimate_v0_from_traj(gt, Lsec)
                        sim = _simulate_single_body(
                            (self.viz.traj_learned_params['restitution'],
                             self.viz.traj_learned_params['lateralFriction'],
                             self.viz.traj_learned_params['mass']),
                            steps=steps, init_height=max(0.05, gt[0, 2]), init_v0_xyz=v0_est, mode=p.DIRECT
                        )
                        sim_rel = sim - sim[0]
                        sim_rel = _downsample_nd(sim_rel, ds)

                        # DTW 가능하면 DTW와 L2 둘 다, 아니면 L2만
                        L_dtw = None
                        try:
                            L_dtw = _traj_loss(gt_rel, sim_rel, use_dtw=True)
                        except Exception:
                            pass
                        L_l2 = _traj_loss(gt_rel, sim_rel, use_dtw=False)

                        gt_params = getattr(self.viz, "traj_gt_params", {})
                        print("[trajectory][train] ---- RESULT ----")
                        print(f"[trajectory][train] GT params      : {gt_params}")
                        print(f"[trajectory][train] Learned params : {self.viz.traj_learned_params}")
                        if L_dtw is not None:
                            print(f"[trajectory][train] Loss (DTW)    : {L_dtw:.6f}")
                        print(f"[trajectory][train] Loss (L2)     : {L_l2:.6f}")
                        print(f"[trajectory][train] Steps/DS      : steps={steps}, downsample={ds}")
                        print("-------------------------------")

                    except Exception as e:
                        print("[trajectory] training error:", e)
                        self.viz.traj_learned_params = {'lateralFriction': 0.3, 'restitution': 0.3}
                    finally:
                        self.viz.traj_training_done = True
                        self._training_in_progress = False

                threading.Thread(target=_worker, daemon=True).start()

            imgui.same_line()
            # Visualize toggle
            changed, toggle = imgui.checkbox("Visualize Trajectory", self.viz.traj_viz_on)
            if changed:
                self.viz.traj_viz_on = bool(toggle)

            # 상태 텍스트
            if self._recording_gt:
                imgui.text(f"Recording GT... {time.time()-self._start_time:.1f}s")
            elif self._recording_pred:
                imgui.text(f"Recording Pred... {time.time()-self._start_time:.1f}s")
            elif self._training_in_progress:
                imgui.text("Training... (background)")
            else:
                if len(self.viz.traj_gt) > 0:
                    imgui.text(f"GT points: {len(self.viz.traj_gt)}")
                if len(self.viz.traj_pred) > 0:
                    imgui.text(f"Pred points: {len(self.viz.traj_pred)}")

        # --- 프레임별 업데이트 (UI 아래) ---
        # 1) 방금 소환된 오브젝트 감지 → GT 녹화 시작
        if self._await_click_spawn and getattr(self.viz, "last_spawned_bullet_id", None) is not None:
            self._await_click_spawn = False
            self._start_gt_recording(self.viz.last_spawned_bullet_id)
            # 소모 후 클리어
            self.viz.last_spawned_bullet_id = None

        # 2) GT 녹화 진행
        if self._recording_gt and self._tracked_bid is not None:
            elapsed = time.time() - self._start_time
            if elapsed >= self.viz.traj_length:
                self._stop_gt_recording()
            else:
                self._maybe_sample_current(self._tracked_bid, self.viz.traj_gt)

        # 3) 학습 완료 → Pred 녹화 시작 트리거
        if getattr(self.viz, "traj_training_done", False) and (not self._recording_pred) and (not self._recording_gt):
            # 한 번만 소비
            self.viz.traj_training_done = False
            self._start_pred_recording()  # ★ _tracked_bid 검사 제거 (함수 내부에서 판단)

        # 4) Pred 녹화 진행
        if self._recording_pred and self._pred_bid is not None:  # ★ pred용 임시 바디 핸들 사용
            elapsed = time.time() - self._start_time
            if elapsed >= self.viz.traj_length:
                self._stop_pred_recording()
            else:
                self._maybe_sample_current(self._pred_bid, self.viz.traj_pred)  # ★ 여기서도 _pred_bid
