# bullet_room_fit.py  (was: warp_room_sim.py)
# ─────────────────────────────────────────────────────────────
# PyBullet + SPSA only
# - Static room mesh (OBJ) vs. dynamic rigid sphere (from OBJ bounding sphere)
# - One-shot run: generate GT by Bullet → fit (SPSA, 2-stage) → overlay/save
# ─────────────────────────────────────────────────────────────

import argparse, os, sys, math, random
from time import perf_counter
import numpy as np
import trimesh
import open3d as o3d

# PyBullet
try:
    import pybullet as p
    import pybullet_data
except Exception as _e:
    raise RuntimeError("PyBullet not available. Install with: pip install pybullet") from _e


# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────

def rotx(deg: float):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)

def load_obj_vertices_faces(path: str):
    mesh = trimesh.load(path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"'{path}' 로드 실패(삼각형 메쉬 아님)")
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int32)
    return v, f, mesh

def compute_bounding_sphere(verts: np.ndarray, center_hint: str = "com"):
    if center_hint == "origin":
        c = np.zeros(3, dtype=np.float32)
    elif center_hint == "bbox":
        bbmin = verts.min(0); bbmax = verts.max(0)
        c = (bbmin + bbmax) * 0.5
    else:  # "com"
        c = verts.mean(0)
    r = np.linalg.norm(verts - c, axis=1).max()
    return c.astype(np.float32), float(r)


# ─────────────────────────────────────────────────────────────
# PyBullet backend
# ─────────────────────────────────────────────────────────────

def build_bullet_cache(scene_obj, object_obj, scene_rotx_deg, start_pos, gravity, dt, substeps, radius_hint=None):
    """Create Bullet world once and reuse."""
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(gravity[0], gravity[1], gravity[2])

    h = float(dt)/float(max(1, substeps))
    p.setTimeStep(h)

    # scene (static concave mesh)
    flags = p.GEOM_FORCE_CONCAVE_TRIMESH
    scene_col = p.createCollisionShape(p.GEOM_MESH, fileName=scene_obj, flags=flags)
    scene_vis = p.createVisualShape(p.GEOM_MESH, fileName=scene_obj)
    rx = math.radians(scene_rotx_deg)
    scene_quat = p.getQuaternionFromEuler([rx, 0.0, 0.0])
    scene_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=scene_col,
                                 baseVisualShapeIndex=scene_vis,
                                 basePosition=[0,0,0], baseOrientation=scene_quat)

    # dynamic sphere (bounding sphere of object)
    ov, of, _ = load_obj_vertices_faces(object_obj)
    _, r_obj = compute_bounding_sphere(ov, center_hint="com")
    r = float(radius_hint if radius_hint is not None else r_obj*1.01)

    ball_col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
    ball_vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[1,0,0,1])
    ball_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=ball_col,
                                baseVisualShapeIndex=ball_vis,
                                basePosition=start_pos, baseOrientation=[0,0,0,1])

    return dict(client=cid, scene_id=scene_id, ball_id=ball_id,
                r=r, h=h, substeps=substeps, start=start_pos, scene_quat=scene_quat)

def simulate_bullet_once(args_base, vx, vy, vz, mu, e, mass_unused=1.0, seconds=None):
    """Generate one trajectory using cached Bullet world. Mass is ignored (single body vs static scene)."""
    cache = args_base["bullet_cache"]
    ball_id = cache["ball_id"]

    p.resetBasePositionAndOrientation(ball_id, cache["start"], [0,0,0,1])
    p.resetBaseVelocity(ball_id, linearVelocity=[vx,vy,vz], angularVelocity=[0,0,0])

    p.changeDynamics(ball_id, -1, lateralFriction=float(mu), restitution=float(e),
                     linearDamping=0.0, angularDamping=0.0)

    seconds = args_base["seconds"] if seconds is None else seconds
    steps = int(seconds/args_base["dt"])
    substeps = cache["substeps"]

    traj = np.zeros((steps+1, 1, 3), dtype=np.float32)
    pos, _ = p.getBasePositionAndOrientation(ball_id)
    traj[0,0,:] = np.array(pos, dtype=np.float32)

    for s in range(1, steps+1):
        for _ in range(substeps):
            p.stepSimulation()
        pos, _ = p.getBasePositionAndOrientation(ball_id)
        traj[s,0,:] = np.array(pos, dtype=np.float32)

    return traj

def bullet_record_traj(scene_path: str,
                       object_path: str,
                       seconds: float,
                       dt: float,
                       substeps: int,
                       scene_rotx_deg: float,
                       gravity=(0.0, -9.81, 0.0),
                       start_pos=(0.0, 1.0, 0.0),
                       v0=(0.0, 0.0, 0.0),
                       mu: float = 0.3,
                       e: float = 0.3,
                       mass_unused: float = 1.0):
    """Build a Bullet world for a single shot and return [T+1,1,3] traj."""
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(*gravity)

    h = float(dt) / max(1, int(substeps))
    p.setTimeStep(h)

    flags = p.GEOM_FORCE_CONCAVE_TRIMESH
    scene_col = p.createCollisionShape(p.GEOM_MESH, fileName=scene_path, flags=flags)
    scene_vis = p.createVisualShape(p.GEOM_MESH, fileName=scene_path)
    rx = math.radians(scene_rotx_deg)
    scene_quat = p.getQuaternionFromEuler([rx, 0.0, 0.0])
    _scene_id = p.createMultiBody(baseMass=0.0,
                                  baseCollisionShapeIndex=scene_col,
                                  baseVisualShapeIndex=scene_vis,
                                  basePosition=[0, 0, 0],
                                  baseOrientation=scene_quat)

    ov, of, _ = load_obj_vertices_faces(object_path)
    _, r_obj = compute_bounding_sphere(ov, center_hint="com")
    r = float(r_obj * 1.01)

    ball_col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
    ball_vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[1, 0, 0, 1])
    ball_id = p.createMultiBody(baseMass=1.0,
                                baseCollisionShapeIndex=ball_col,
                                baseVisualShapeIndex=ball_vis,
                                basePosition=list(start_pos),
                                baseOrientation=[0, 0, 0, 1])

    p.changeDynamics(ball_id, -1, lateralFriction=float(mu), restitution=float(e),
                     linearDamping=0.0, angularDamping=0.0)
    p.resetBaseVelocity(ball_id, list(v0), [0, 0, 0])

    steps = int(seconds / dt)
    traj = np.zeros((steps + 1, 1, 3), dtype=np.float32)
    pos, _ = p.getBasePositionAndOrientation(ball_id)
    traj[0, 0, :] = np.asarray(pos, dtype=np.float32)

    for s in range(1, steps + 1):
        for _ in range(substeps):
            p.stepSimulation()
        pos, _ = p.getBasePositionAndOrientation(ball_id)
        traj[s, 0, :] = np.asarray(pos, dtype=np.float32)

    p.disconnect()
    return traj


# ─────────────────────────────────────────────────────────────
# Loss / Opt / Viz
# ─────────────────────────────────────────────────────────────

def mse_loss(traj_pred: np.ndarray, traj_gt: np.ndarray, stride: int = 1, window_steps: int = None):
    P = traj_pred[:, 0, :] if traj_pred.ndim == 3 else traj_pred
    G = traj_gt[:, 0, :]    if traj_gt.ndim == 3 else traj_gt
    T = min(len(P), len(G))
    if window_steps is not None:
        T = min(T, window_steps)
    P = P[:T:stride]; G = G[:T:stride]
    d = P - G
    return float(np.mean(np.sum(d*d, axis=1)))

def estimate_v0_from_gt(gt: np.ndarray, dt: float, gvec=(0., -9.81, 0.)):
    p0 = gt[0, 0, :] if gt.ndim == 3 else gt[0]
    p1 = gt[1, 0, :] if gt.ndim == 3 else gt[1]
    g = np.array(gvec, dtype=np.float32)
    return (p1 - p0)/dt - 0.5*g*dt

def clamp_params(theta, bounds, v_bounds=None):
    """theta = [vx, vy, vz, mu, e, mass_dummy]"""
    vx, vy, vz, mu, e, m = theta
    (mu_min, mu_max), (e_min, e_max), _m_bounds = bounds
    if v_bounds is not None:
        (vx_min, vx_max), (vy_min, vy_max), (vz_min, vz_max) = v_bounds
        vx = float(np.clip(vx, vx_min, vx_max))
        vy = float(np.clip(vy, vy_min, vy_max))
        vz = float(np.clip(vz, vz_min, vz_max))
    mu = float(np.clip(mu, mu_min, mu_max))
    e  = float(np.clip(e,  e_min,  e_max))
    # mass_dummy ignored
    return np.array([vx, vy, vz, mu, e, m], dtype=np.float32)

def spsa_optimize(gt_traj, args_base,
                  theta0,
                  iters=12,
                  a0=0.08, c0=0.15, Ak=3.0, alpha=0.602, gamma=0.101,
                  stride=2,
                  fast_seconds=0.5,
                  store_k=10,
                  update_scale=None,
                  v_bounds=None):
    """SPSA with cached Bullet world."""
    rng = random.Random(0xC0FFEE)
    theta = clamp_params(theta0.copy(),
                         bounds=((0.0,1.2),(0.0,0.9),(0.0,10.0)),
                         v_bounds=v_bounds)
    upd = np.ones(6, dtype=np.float32) if update_scale is None else np.array(update_scale, dtype=np.float32)
    hist = []
    T_fast = fast_seconds

    # warmup
    _ = simulate_bullet_once(args_base, *theta[:3], theta[3], theta[4], theta[5], seconds=T_fast)

    for k in range(iters):
        ck = c0 / ((k+1.0)**gamma)
        ak = a0 / ((k+1.0+Ak)**alpha)
        delta = np.array([1 if rng.random()<0.5 else -1 for _ in range(6)], dtype=np.float32)

        thetap = clamp_params(theta + ck*delta,
                              bounds=((0.0,1.2),(0.0,0.9),(0.0,10.0)),
                              v_bounds=v_bounds)
        thetam = clamp_params(theta - ck*delta,
                              bounds=((0.0,1.2),(0.0,0.9),(0.0,10.0)),
                              v_bounds=v_bounds)

        traj_p = simulate_bullet_once(args_base, *thetap[:3], thetap[3], thetap[4], thetap[5], seconds=T_fast)
        traj_m = simulate_bullet_once(args_base, *thetam[:3], thetam[3], thetam[4], thetam[5], seconds=T_fast)

        fp = mse_loss(traj_p, gt_traj, stride=stride)
        fm = mse_loss(traj_m, gt_traj, stride=stride)

        ghat = (fp - fm)/(2.0*ck) * (1.0/delta)
        theta = clamp_params(theta - ak*(upd*ghat),
                             bounds=((0.0,1.2),(0.0,0.9),(0.0,10.0)),
                             v_bounds=v_bounds)

        if len(hist) < store_k:
            traj_now = simulate_bullet_once(args_base, *theta[:3], theta[3], theta[4], theta[5], seconds=T_fast)
            hist.append((k, traj_now))

    traj_best = simulate_bullet_once(args_base, *theta[:3], theta[3], theta[4], theta[5], seconds=args_base["seconds"])
    final_loss = mse_loss(traj_best, gt_traj, stride=stride)
    return theta, traj_best, hist, final_loss

def draw_overlay_o3d(sv, sf, gt_traj, best_traj, hist_list, radii):
    scene = o3d.geometry.TriangleMesh()
    scene.vertices = o3d.utility.Vector3dVector(sv.astype(np.float64))
    scene.triangles = o3d.utility.Vector3iVector(sf.astype(np.int32))
    scene.compute_vertex_normals()
    scene.paint_uniform_color([0.75, 0.75, 0.75])

    geoms = [scene]

    def make_lines(pts, color):
        pts = np.asarray(pts)
        lines = np.stack([np.arange(len(pts)-1), np.arange(1,len(pts))], axis=1)
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines)
        )
        ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines),1)))
        return ls

    gt_pts = gt_traj[:,0,:] if gt_traj.ndim==3 else gt_traj
    geoms.append(make_lines(gt_pts, np.array([0.0, 0.3, 1.0])))

    if len(hist_list) > 0:
        colors_hist = [np.array([1.0, 0.85 - 0.85*i/max(1, len(hist_list)-1), 0.0]) for i in range(len(hist_list))]
        for (k, t) in hist_list:
            pts = t[:,0,:] if t.ndim==3 else t
            geoms.append(make_lines(pts, colors_hist[min(k, len(colors_hist)-1)]))

    best_pts = best_traj[:,0,:] if best_traj.ndim==3 else best_traj
    geoms.append(make_lines(best_pts, np.array([0.0, 0.9, 0.2])))

    sp = o3d.geometry.TriangleMesh.create_sphere(radius=float(radii[0]))
    sp.translate(best_pts[0].astype(np.float64))
    sp.paint_uniform_color([1.0, 0.0, 0.0])
    geoms.append(sp)

    o3d.visualization.draw_geometries(geoms)


# ─────────────────────────────────────────────────────────────
# Main (default: generate GT → fit, Bullet only)
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, required=True)
    ap.add_argument("--objects", type=str, nargs="+", required=True)  # first is used
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--substeps", type=int, default=6)
    ap.add_argument("--scene-rotx", type=float, default=0.0)
    ap.add_argument("--gravity", type=float, nargs=3, default=[0.0, -9.81, 0.0])

    ap.add_argument("--starts", type=float, nargs="*", default=None, help="x y z (single body)")
    ap.add_argument("--v0s", type=float, nargs="*", default=None, help="initial guess for v0 (x y z)")

    # GT parameters for on-the-fly generation (Bullet)
    ap.add_argument("--gt-v0s", type=float, nargs="*", default=None, help="GT v0 (x y z)")
    ap.add_argument("--gt-mus", type=float, nargs="*", default=None, help="GT mu")
    ap.add_argument("--gt-es", type=float, nargs="*", default=None, help="GT e")
    ap.add_argument("--gt-save", type=str, default=None, help="optional: save generated GT traj .npy")

    # Fitting hyper-params (2-stage by default)
    ap.add_argument("--fit-2stage", action="store_true", default=True)
    ap.add_argument("--fit-iters1", type=int, default=6)
    ap.add_argument("--fit-iters2", type=int, default=6)
    ap.add_argument("--fit-fast-seconds1", type=float, default=0.5)
    ap.add_argument("--fit-fast-seconds2", type=float, default=None, help="None → full seconds")
    ap.add_argument("--fit-stride1", type=int, default=3)
    ap.add_argument("--fit-stride2", type=int, default=1)
    ap.add_argument("--fit-vmax", type=float, default=12.0, help="|vx,vy,vz| clamp")

    ap.add_argument("--viz-samples", type=int, default=10)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--save", type=str, default=None, help="save final predicted traj .npy")

    args = ap.parse_args()

    assert len(args.objects) >= 1, "at least one dynamic object required"
    obj_path = args.objects[0]

    # Start position
    if args.starts is not None:
        start = np.array(args.starts, dtype=np.float32).reshape(-1,3)[0].tolist()
    else:
        start = [0.0, 1.0, 0.0]

    # (A) Generate GT by Bullet in this run
    gt_v0 = (np.array(args.gt_v0s, dtype=np.float32).reshape(-1,3)[0].tolist()
             if args.gt_v0s is not None else [0.0, 0.0, 0.0])
    gt_mu = (float(args.gt_mus[0]) if args.gt_mus is not None else 0.3)
    gt_e  = (float(args.gt_es[0])  if args.gt_es  is not None else 0.3)

    gt = bullet_record_traj(
        scene_path=args.scene,
        object_path=obj_path,
        seconds=args.seconds, dt=args.dt, substeps=args.substeps,
        scene_rotx_deg=args.scene_rotx,
        gravity=tuple(args.gravity),
        start_pos=start, v0=gt_v0, mu=gt_mu, e=gt_e, mass_unused=1.0
    )
    if args.gt_save:
        np.save(args.gt_save, gt)
        print(f"[GT] saved to {args.gt_save} shape={gt.shape}")

    # (B) Build reusable Bullet world for fast fitting
    args_base = dict(
        seconds=args.seconds, dt=args.dt, substeps=args.substeps,
        scene=args.scene, obj=obj_path,
        scene_rotx=args.scene_rotx, gravity=tuple(args.gravity),
        start=start
    )
    args_base["bullet_cache"] = build_bullet_cache(
        scene_obj=args.scene,
        object_obj=obj_path,
        scene_rotx_deg=args.scene_rotx,
        start_pos=start,
        gravity=tuple(args.gravity),
        dt=args.dt,
        substeps=args.substeps,
        radius_hint=None
    )

    # (C) Initial theta: v0 from GT (if no initial guess), μ/e from args or defaults, mass dummy
    if args.v0s is not None:
        v0_init = np.array(args.v0s, dtype=np.float32).reshape(-1,3)[0]
    else:
        v0_init = estimate_v0_from_gt(gt, args.dt, gvec=args_base["gravity"])

    mu_init = 0.3
    e_init  = 0.3
    m_init  = 1.0  # dummy, frozen
    theta0  = np.array([v0_init[0], v0_init[1], v0_init[2], mu_init, e_init, m_init], dtype=np.float32)

    v_bounds = ((-args.fit_vmax, args.fit_vmax),
                (-args.fit_vmax, args.fit_vmax),
                (-args.fit_vmax, args.fit_vmax))

    # (D) Fit (2-stage default)
    t0 = perf_counter()
    if args.fit_2stage:
        # Stage-1: optimize v only on short window
        theta1, traj1, hist1, loss1 = spsa_optimize(
            gt_traj=gt, args_base=args_base, theta0=theta0,
            iters=args.fit_iters1, a0=0.08, c0=0.15, Ak=3.0,
            stride=args.fit_stride1, fast_seconds=args.fit_fast_seconds1,
            store_k=max(1, args.viz_samples//2),
            update_scale=[1.,1.,1., 0.,0., 0.],
            v_bounds=v_bounds
        )
        # Stage-2: freeze v, optimize μ/e on long window
        fs2 = args.fit_fast_seconds2 if args.fit_fast_seconds2 is not None else args.seconds
        theta, traj_best, hist2, final_loss = spsa_optimize(
            gt_traj=gt, args_base=args_base, theta0=theta1,
            iters=args.fit_iters2, a0=0.08, c0=0.15, Ak=3.0,
            stride=args.fit_stride2, fast_seconds=fs2,
            store_k=args.viz_samples,
            update_scale=[0.,0.,0., 1.,1., 0.],
            v_bounds=v_bounds
        )
        # merge history
        hist = []
        take1 = min(len(hist1), max(1, args.viz_samples//2))
        hist.extend(hist1[:take1])
        take2 = min(len(hist2), args.viz_samples - take1)
        hist.extend(hist2[:take2])
    else:
        theta, traj_best, hist, final_loss = spsa_optimize(
            gt_traj=gt, args_base=args_base, theta0=theta0,
            iters=(args.fit_iters1+args.fit_iters2),
            a0=0.08, c0=0.15, Ak=3.0,
            stride=2, fast_seconds=0.5,
            store_k=args.viz_samples,
            update_scale=[1.,1.,1., 1.,1., 0.],
            v_bounds=v_bounds
        )
    t1 = perf_counter()

    print(f"[FIT] done in {t1 - t0:.3f}s")
    print(f"[FIT] params: v0=({theta[0]:.3f},{theta[1]:.3f},{theta[2]:.3f}), mu={theta[3]:.3f}, e={theta[4]:.3f}")
    print(f"[FIT] loss={final_loss:.6f}")

    # (E) Overlay
    if not args.no_preview:
        sv, sf, _ = load_obj_vertices_faces(args.scene)
        if abs(args.scene_rotx) > 1e-6:
            R = rotx(args.scene_rotx); sv = (sv @ R.T).astype(np.float32)
        ov, of, _ = load_obj_vertices_faces(obj_path)
        _, r0 = compute_bounding_sphere(ov, center_hint="com")
        draw_overlay_o3d(sv, sf, gt, traj_best, hist, [r0*1.01])

    if args.save:
        np.save(args.save, traj_best)
        print(f"[saved] {args.save} shape={traj_best.shape}")

    # disconnect Bullet cache
    p.disconnect(args_base["bullet_cache"]["client"])

if __name__ == "__main__":
    main()
