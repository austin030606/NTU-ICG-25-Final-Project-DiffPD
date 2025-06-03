import sys
sys.path.append('../')

import time
from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt

from py_diff_pd.common.common import create_folder, print_info
from py_diff_pd.env.armadillo_env_3d import ArmadilloEnv3d

def run_experiments():
    # 0) Setup
    seed = 42
    out  = Path("armadillo_combined_experiment")
    create_folder(out)

    # Env config
    common_cfg = {
        'youngs_modulus': 5e5,
        'init_rotate_angle': 0,
        'state_force_parameters': [0, 0, 0],
        'spp': 4,
        'mesh_file': "../../asset/mesh/armadillo_high_res.obj"
    }
    env = ArmadilloEnv3d(seed, out, common_cfg)

    # Multi‐frame + oscillating boundary force
    deformable = env.deformable()
    q0 = env.default_init_position()
    v0 = env.default_init_velocity()
    frame_num = 30
    a0 = np.zeros(deformable.act_dofs())
    # build f0 like before…
    f0 = []
    for t in range(frame_num):
        θ = -np.pi/4 + (t/frame_num)* (np.pi/2)
        Fmag = t/frame_num
        direction = np.array([np.cos(θ), np.sin(θ), 0]) * Fmag
        fi = np.zeros(deformable.dofs())
        for i in env.min_x_nodes(): fi[3*i:3*i+3] =  direction
        for i in env.max_x_nodes(): fi[3*i:3*i+3] = -direction
        f0.append(fi)
    a_list = [a0]*frame_num

    # Solver settings
    method = 'pd_eigen_pcg'   # or 'newton_cholesky'
    opts_base = {
        'max_pd_iter': 5000,
        'max_ls_iter': 10,
        'abs_tol': 1e-9,
        'rel_tol': 1e-4,
        'verbose': 0,
        'thread_ct': 4,
        'use_bfgs': 1,
        'bfgs_history_size': 10
    }

    # Experiment: Tolerance → Runtime & Energy Drift
    # tolerances = [(1e-4,1e-6), (1e-5,1e-6), (1e-5,1e-9)]
    rel_list = [1e-3, 1e-4, 1e-5]
    abs_list = [1e-6, 1e-8, 1e-10]
    tolerances = [(r, a) for r in rel_list for a in abs_list]
    
    labels     = [f"rel={r:.0e}\nabs={a:.0e}" for r,a in tolerances]
    metrics    = {'runtime': [], 'drift': []}

    # 2a) reference energy @ first tolerance
    ref_rel, ref_abs = tolerances[0]
    ref_cfg = opts_base.copy()
    ref_cfg.update({'rel_tol': ref_rel, 'abs_tol': ref_abs})
    print_info("=== computing reference energy ===")
    loss_ref, grad_ref, info_ref = env.simulate(
        1e-3, frame_num, method, ref_cfg,
        q0, v0, a_list, f0,
        require_grad=True, vis_folder=None
    )
    print_info(f" Reference energy = {loss_ref:.6f}")

    # 2b) sweep tolerances
    print_info("=== tol vs metrics ===")
    for rel, abs_ in tolerances:
        cfg = opts_base.copy()
        cfg.update({'rel_tol': rel, 'abs_tol': abs_})
        print_info(f" tol (rel={rel:.0e}, abs={abs_:.0e})")
        t0 = time.time()
        loss, grad, info = env.simulate(
            1e-3, frame_num, method, cfg,
            q0, v0, a_list, f0,
            require_grad=True, vis_folder=None
        )
        runtime = time.time() - t0
        drift   = loss - loss_ref
        metrics['runtime'].append(runtime)
        metrics['drift'].append(drift)
        print_info(f"   time={runtime:.3f}s, drift={drift:.3e}")

    # save & plot
    with open(out/'tol_metrics.pkl','wb') as f:
        pickle.dump((tolerances, metrics), f)

    x = np.arange(len(tolerances))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10,8))

    # Left axis: Energy drift
    bars1 = ax1.bar(x - width/2,
                    metrics['drift'],
                    width,
                    label='Energy drift')
    ax1.set_ylabel('Energy drift')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis='y', linestyle='--')

    # Label energy‐drift bars
    for rect in bars1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2, h, f"{h:.3e}",
                ha="center", va="bottom", fontsize=9)

    # Right axis: Runtime
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2,
                    metrics['runtime'],
                    width,
                    label='Runtime (s)',
                    alpha=0.6)
    ax2.set_ylabel('Runtime (s)')

    # Label runtime bars
    for rect in bars2:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2, h, f"{h:.1f}",
                ha="center", va="bottom", fontsize=9)

    # Combined legend
    ax1.legend(
        [bars1, bars2],
        ['Energy drift', 'Runtime (s)'],
        loc='upper right',
        frameon=False
    )

    plt.title("Tolerance vs Energy Drift & Runtime")
    plt.tight_layout()
    plt.savefig(out/'tol_vs_metrics.png', dpi=300)
    plt.close()

    print_info(f"All results saved in {out}")

if __name__ == '__main__':
    run_experiments()
