# nimble_obj_fit.py
# -*- coding: utf-8 -*-
import os, math, tempfile, textwrap
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import nimblephysics as nimble

DEBUG = True  # 디버그 로그 on/off

# ---------------------- quat(x,y,z,w) -> rpy ----------------------
def _quat_xyzw_to_rpy(q):
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    t0 = +2.0*(w*x + y*z); t1 = +1.0 - 2.0*(x*x + y*y)
    roll = math.atan2(t0, t1)
    t2 = +2.0*(w*y - z*x); t2 = +1.0 if t2 > +1.0 else t2; t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0*(w*z + x*y); t4 = +1.0 - 2.0*(y*y + z*z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw

def quat_xyzw_to_expmap(q):
    x,y,z,w = q
    n = math.sqrt(x*x+y*y+z*z+w*w) or 1.0
    x,y,z,w = x/n, y/n, z/n, w/n
    ang = 2.0*math.acos(max(min(w,1.0), -1.0))
    s = math.sqrt(max(1.0 - w*w, 0.0))
    if s < 1e-8 or ang < 1e-8: return (0.0,0.0,0.0)
    ax, ay, az = x/s, y/s, z/s
    return (ax*ang, ay*ang, az*ang)

# ---------------------- OBJ bounds & inertia (box approx) ----------------------
def _read_obj_bounds(obj_path: str, scale: Tuple[float,float,float]) -> Tuple[np.ndarray, np.ndarray]:
    s = np.array(scale, dtype=np.float64)
    vmin = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    vmax = -vmin
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z, *rest = line.strip().split()
                v = np.array([float(x), float(y), float(z)], dtype=np.float64)*s
                vmin = np.minimum(vmin, v); vmax = np.maximum(vmax, v)
    if not np.all(np.isfinite(vmin)):
        vmin = np.array([-0.5,-0.5,-0.5])*s; vmax = -vmin
    return vmin, vmax

def _approx_box_inertia_from_bounds(mass: float, vmin: np.ndarray, vmax: np.ndarray) -> Tuple[float,...]:
    d = np.maximum(vmax - vmin, 1e-6); x, y, z = d.tolist()
    Ixx = (1.0/12.0) * mass * (y*y + z*z)
    Iyy = (1.0/12.0) * mass * (x*x + z*z)
    Izz = (1.0/12.0) * mass * (x*x + y*y)
    return (Ixx, 0.0, 0.0, 0.0, Iyy, 0.0, 0.0, 0.0, Izz)

def _world_aabb(obj_path, scale, pos, quat_xyzw):
    # OBJ AABB(로컬) -> 8 corner -> 월드 변환
    vmin, vmax = _read_obj_bounds(obj_path, scale)
    corners = np.array([[vmin[0], vmin[1], vmin[2]],
                        [vmin[0], vmin[1], vmax[2]],
                        [vmin[0], vmax[1], vmin[2]],
                        [vmin[0], vmax[1], vmax[2]],
                        [vmax[0], vmin[1], vmin[2]],
                        [vmax[0], vmin[1], vmax[2]],
                        [vmax[0], vmax[1], vmin[2]],
                        [vmax[0], vmax[1], vmax[2]]], dtype=np.float64)
    # R from quat(x,y,z,w)
    x,y,z,w = quat_xyzw
    # 회전행렬
    xx,yy,zz = x*x,y*y,z*z; xy,xz,yz = x*y,x*z,y*z; wx,wy,wz = w*x,w*y,w*z
    R = np.array([[1-2*(yy+zz), 2*(xy-wz),     2*(xz+wy)],
                  [2*(xy+wz),   1-2*(xx+zz),   2*(yz-wx)],
                  [2*(xz-wy),   2*(yz+wx),     1-2*(xx+yy)]], dtype=np.float64)
    t = np.asarray(pos, np.float64)
    P = (R @ corners.T).T + t
    return P.min(axis=0), P.max(axis=0)

# ---------------------- URDF builder for an OBJ mesh ----------------------
def build_urdf_for_obj(
    name: str,
    obj_path: str,
    mass: float = 1.0,
    scale: Tuple[float,float,float] = (1.0,1.0,1.0),
    base_pos_xyz: Tuple[float,float,float] = (0.0,0.0,0.0),
    base_quat_xyzw: Tuple[float,float,float,float] = (0.0,0.0,0.0,1.0),
    dynamic: bool = True,
) -> str:
    obj_abs = os.path.abspath(obj_path)
    if not os.path.exists(obj_abs):
        raise FileNotFoundError(f"OBJ not found: {obj_abs}")

    vmin, vmax = _read_obj_bounds(obj_abs, scale)
    mass = float(max(mass, 1e-6)) if dynamic else 1.0
    I = _approx_box_inertia_from_bounds(mass, vmin, vmax)

    sx, sy, sz = scale
    inertial_xml = ""
    if dynamic:
        inertial_xml = f"""
      <inertial>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <mass value="{mass:.9f}"/>
        <inertia ixx="{I[0]:.9e}" ixy="{I[1]:.9e}" ixz="{I[2]:.9e}"
                 iyy="{I[4]:.9e}" iyz="{I[5]:.9e}" izz="{I[8]:.9e}"/>
      </inertial>
    """

    # ⚠️ DART는 file:// prefix가 더 안전
    mesh_path = "file://" + obj_abs

    link_block = f"""
    <link name="{name}_link">
      {inertial_xml}
      <visual>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry><mesh filename="{mesh_path}" scale="{sx} {sy} {sz}"/></geometry>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry><mesh filename="{mesh_path}" scale="{sx} {sy} {sz}"/></geometry>
      </collision>
    </link>
    """

    if dynamic:
        base_link_block = f"""<link name="{name}_base"/>"""
        joint_block = f"""
    <joint name="{name}_root" type="floating">
      <parent link="{name}_base"/>
      <child link="{name}_link"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </joint>"""
    else:
        bx, by, bz = base_pos_xyz
        rx, ry, rz = _quat_xyzw_to_rpy(base_quat_xyzw)
        base_link_block = f"""<link name="{name}_base"/>"""
        joint_block = f"""
    <joint name="{name}_weld" type="fixed">
      <parent link="{name}_base"/>
      <child  link="{name}_link"/>
      <origin xyz="{bx} {by} {bz}" rpy="{rx} {ry} {rz}"/>
    </joint>"""

    urdf = f"""<?xml version="1.0"?>
<robot name="{name}">
{base_link_block}
{link_block}
{joint_block}
</robot>"""
    return textwrap.dedent(urdf)

# ---------------------- helpers ----------------------
def _apply_material_to_skeleton(skel, mu: float, restitution: float):
    try: n = skel.getNumBodyNodes()
    except Exception: n = 0
    for i in range(n):
        bn = skel.getBodyNode(i)
        try: bn.setFrictionCoeff(float(mu))
        except Exception: pass
        try: bn.setRestitutionCoeff(float(restitution))
        except Exception: pass

def _set_mass_on_skeleton_root(skel, mass: float):
    try:
        skel.getBodyNode(0).setMass(float(mass))
    except Exception:
        pass

def load_mesh_as_skeleton(
    world, name, obj_path, mass, mu, restitution,
    scale=(1.0,1.0,1.0), base_pos_xyz=(0,0,0),
    base_quat_xyzw=(0,0,0,1), dynamic=True,
):
    urdf = build_urdf_for_obj(
        name=name,
        obj_path=obj_path,
        mass=mass,
        scale=scale,
        base_pos_xyz=base_pos_xyz,
        base_quat_xyzw=base_quat_xyzw,
        dynamic=dynamic,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(urdf)
        urdf_path = f.name
    skel = world.loadSkeleton(urdf_path)
    _apply_material_to_skeleton(skel, mu, restitution)
    if dynamic:
        _set_mass_on_skeleton_root(skel, mass)
    return skel

def _try_enable_fcl(world):
    # 빌드에 따라 노출되지 않을 수 있음 → 실패해도 무시
    try:
        from nimblephysics import collision
        det = collision.FCLCollisionDetector()
        world.getConstraintSolver().setCollisionDetector(det)
        if DEBUG: print("[nimble] collision detector = FCL")
    except Exception as e:
        if DEBUG: print("[nimble] FCL not available:", e)

# ---------------------- rollout ----------------------
def _rollout(world, state0: torch.Tensor, steps: int, dof: int, log_contacts: bool=False):
    pos_traj = []
    state = state0
    zero_ctrl = torch.zeros(dof, dtype=state0.dtype, device=state0.device)
    for t in range(steps):
        state = nimble.timestep(world, state, zero_ctrl)
        q = state[:dof]
        pos_traj.append(q[3:6].clone())
        if log_contacts and (t % 10 == 0):
            try:
                res = world.getLastCollisionResult()
                print(f"[nimble] step {t}: contacts={res.getNumContacts()}")
            except Exception:
                pass
    return torch.stack(pos_traj)

# ---------------------- main fitting ----------------------
def fit_params_from_gt(
    gt_xyz: np.ndarray,
    dt: float,
    obj_path: str,
    init_pose_xyz: Optional[Tuple[float,float,float]] = None,
    init_quat_xyzw: Tuple[float,float,float,float] = (0,0,0,1),
    scene_obj: Optional[str] = None,
    scene_pos_xyz: Tuple[float,float,float] = (0.0,0.0,0.0),
    scene_quat_xyzw: Tuple[float,float,float,float] = (0,0,0,1),
    use_plane: bool = True,
    iters: int = 150,
    record_every: int = 15,
    seed: int = 1234,
    init_guess: Optional[Dict[str, float]] = None,
):
    torch.manual_seed(seed)
    gt = torch.from_numpy(np.asarray(gt_xyz, dtype=np.float32))
    T = gt.shape[0]; assert gt.shape[1] == 3; assert dt > 0

    world = nimble.simulation.World()
    world.setGravity([0, 0, -9.81])   # Bullet과 동일(Z-down)
    world.setTimeStep(float(dt))
    _try_enable_fcl(world)

    if use_plane:
        ground = nimble.dynamics.Skeleton()
        _, body = ground.createWeldJointAndBodyNodePair()
        node = body.createShapeNode(nimble.dynamics.BoxShape([1000.0, 0.1, 1000.0]))
        node.createCollisionAspect(); node.createVisualAspect()
        world.addSkeleton(ground)

    # 정적 scene mesh
    if scene_obj is not None and os.path.exists(scene_obj):
        if DEBUG: print("[nimble] using scene mesh:", scene_obj)
        scene_skel = load_mesh_as_skeleton(
            world, name="static_scene",
            obj_path=scene_obj, mass=1.0, mu=1.0, restitution=0.0,
            scale=(1,1,1),
            base_pos_xyz=scene_pos_xyz, base_quat_xyzw=scene_quat_xyzw,
            dynamic=False
        )
        if DEBUG:
            a, b = _world_aabb(scene_obj, (1,1,1), scene_pos_xyz, scene_quat_xyzw)
            print("[nimble] scene AABB world:", a, b)
    else:
        if DEBUG: print("[nimble] scene mesh not found → plane only")

    # 동적 물체(OBJ 그대로)
    start_xyz = init_pose_xyz if init_pose_xyz is not None else tuple(gt[0].tolist())
    m0 = (init_guess.get("mass") if init_guess else 1.0)
    mu0 = (init_guess.get("mu") if init_guess else 0.3)
    e0  = (init_guess.get("e")  if init_guess else 0.1)

    dyn = load_mesh_as_skeleton(
        world, name="dyn",
        obj_path=obj_path, mass=max(1e-3, float(m0)),
        mu=float(mu0), restitution=float(e0),
        scale=(1,1,1),
        base_pos_xyz=tuple(map(float, start_xyz)),
        base_quat_xyzw=init_quat_xyzw, dynamic=True
    )

    dof, state_size = world.getNumDofs(), world.getStateSize()

    # 파라미터
    v0 = torch.tensor([
        (init_guess.get("vx") if init_guess and "vx" in init_guess else 0.0),
        (init_guess.get("vy") if init_guess and "vy" in init_guess else 0.0),
        (init_guess.get("vz") if init_guess and "vz" in init_guess else 0.0)
    ], dtype=torch.float32, requires_grad=True)
    raw_mass = torch.tensor([math.log(math.exp(m0)-1.0) if m0>0 else 0.0], dtype=torch.float32, requires_grad=True)
    raw_mu   = torch.tensor([math.log(math.exp(mu0)-1.0) if mu0>0 else 0.0], dtype=torch.float32, requires_grad=True)
    raw_e    = torch.tensor([math.atanh(max(min(e0,0.999),-0.999))], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([v0, raw_mass, raw_mu, raw_e], lr=0.15)

    # 초기 state (자세=expmap, 위치=xyz)
    state0 = torch.zeros((state_size,), dtype=torch.float32)
    state0[0:3] = torch.tensor(quat_xyzw_to_expmap(init_quat_xyzw), dtype=torch.float32)
    state0[3:6] = torch.tensor(start_xyz, dtype=torch.float32)
    state0[dof+0:dof+3] = 0.0  # dq_rot = 0

    # 디버그: 궤적 AABB
    if DEBUG:
        a, b = np.min(gt_xyz, axis=0), np.max(gt_xyz, axis=0)
        print("[nimble] GT  AABB world:", a, b)

    snapshots: List[np.ndarray] = []
    pred = None

    for it in range(iters):
        opt.zero_grad()

        m  = torch.nn.functional.softplus(raw_mass)[0] + 1e-6
        mu = torch.nn.functional.softplus(raw_mu)[0]
        e  = (torch.tanh(raw_e)[0] * 0.5 + 0.5).clamp(0.0, 1.0)

        _set_mass_on_skeleton_root(dyn, float(m.item()))
        _apply_material_to_skeleton(dyn, float(mu.item()), float(e.item()))

        s = state0.clone()
        s[dof+3:dof+6] = v0  # dq_trans = v0
        pred = _rollout(world, s, steps=T, dof=dof, log_contacts=(DEBUG and it==0))

        loss = torch.mean((pred - gt)**2)
        loss.backward(); opt.step()

        if (it % record_every) == 0 or (it == iters-1):
            snapshots.append(pred.detach().cpu().numpy())

    est = {
        "mass": float(torch.nn.functional.softplus(raw_mass)[0].item()),
        "mu":   float(torch.nn.functional.softplus(raw_mu)[0].item()),
        "e":    float((torch.tanh(raw_e)[0]*0.5+0.5).item()),
        "v0x":  float(v0[0].item()),
        "v0y":  float(v0[1].item()),
        "v0z":  float(v0[2].item()),
    }
    return est, pred.detach().cpu().numpy(), snapshots
