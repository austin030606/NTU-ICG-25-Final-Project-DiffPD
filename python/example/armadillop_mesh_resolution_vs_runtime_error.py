import sys
sys.path.append('../')

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from py_diff_pd.common.common import create_folder, print_info
from py_diff_pd.env.armadillo_env_3d import ArmadilloEnv3d

def sweep_mesh_resolution():
    # --- 输出目录
    out = Path('armadillo_mesh_sweep')
    create_folder(out)

    # --- 网格列表 & 对应元素数
    mesh_paths = [
        '/home/akinesia112/diff_pd_public/asset/mesh/armadillo_low_res.obj',
        '/home/akinesia112/diff_pd_public/asset/mesh/armadillo_mid1.obj',
        '/home/akinesia112/diff_pd_public/asset/mesh/armadillo_mid2.obj',
        '/home/akinesia112/diff_pd_public/asset/mesh/armadillo_high_res.obj',
    ]
    element_counts = [4022, 7681, 11340, 15000]

    # --- 统一物理／求解设置
    seed = 42
    common_kwargs = {
        'youngs_modulus': 5e5,
        'init_rotate_angle': 0,
        'state_force_parameters': [0, 0, 0],
        'spp': 4
    }
    dt = 1e-3            # 固定时间步长
    frame_num = 1        # 只算一帧
    method = 'newton_cholesky'
    opts = {
        'max_newton_iter': 5000,
        'max_ls_iter': 10,
        'abs_tol': 1e-9,
        'rel_tol': 1e-4,
        'verbose': 0,
        'thread_ct': 4
    }

    runtimes = []
    energies = []

    # --- 第一次先跑最高精度网格，得到参考能量
    ref_mesh = mesh_paths[-1]
    print_info(f"Reference run on {Path(ref_mesh).name}")
    cfg = dict(common_kwargs)
    cfg['mesh_file'] = ref_mesh    # 加到 config 里
    env = ArmadilloEnv3d(seed, out, cfg)
    
    deformable = env.deformable()
    q0 = env.default_init_position()
    v0 = env.default_init_velocity()
    a0 = np.zeros(deformable.act_dofs())
    f0 = [np.zeros(deformable.dofs())]
    t0 = time.time()
    loss_ref, info = env.simulate(
        dt, frame_num, method, opts,
        q0, v0, [a0], f0,
        require_grad=False, vis_folder=None
    )
    t_ref = time.time() - t0
    print_info(f"  ref energy = {loss_ref:.6f}, time = {t_ref:.3f}s")

    # --- 对每个网格做扫描
    for mesh_path, n_elem in zip(mesh_paths, element_counts):
        print_info(f"Run on {Path(mesh_path).name} ({n_elem} elems)")
        cfg = dict(common_kwargs)
        cfg['mesh'] = ref_mesh
        env = ArmadilloEnv3d(seed, out, cfg)
        deformable = env.deformable()
        q0 = env.default_init_position()
        v0 = env.default_init_velocity()
        a0 = np.zeros(deformable.act_dofs())
        f0 = [np.zeros(deformable.dofs())]

        t0 = time.time()
        loss, info = env.simulate(
            dt, frame_num, method, opts,
            q0, v0, [a0], f0,
            require_grad=False, vis_folder=None
        )
        t_elapsed = time.time() - t0
        runtimes.append(t_elapsed)
        energies.append(loss)

        err = abs(loss - loss_ref)
        print_info(f"  time = {t_elapsed:.3f}s, energy = {loss:.6f}, error = {err:.3e}")

    # --- 保存数据
    np.save(out / 'element_counts.npy', np.array(element_counts))
    np.save(out / 'runtimes.npy', np.array(runtimes))
    np.save(out / 'energies.npy', np.array(energies))
    np.save(out / 'energy_error.npy', np.abs(np.array(energies) - loss_ref))

    # --- 绘图
    plt.figure()
    plt.plot(element_counts, runtimes, 'o-', label='Runtime (s)')
    plt.xlabel('Element Count')
    plt.ylabel('Time (s)')
    plt.title('Runtime vs Mesh Resolution')
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(out / 'runtime_vs_resolution.png', dpi=300)
    plt.close()

    plt.figure()
    plt.plot(element_counts, np.abs(np.array(energies) - loss_ref), 's-', label='Energy Error')
    plt.xlabel('Element Count')
    plt.ylabel('Error in Energy')
    plt.title('Energy Error vs Mesh Resolution')
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(out / 'error_vs_resolution.png', dpi=300)
    plt.close()

    print_info(f"All results saved in {out}")

if __name__ == '__main__':
    sweep_mesh_resolution()