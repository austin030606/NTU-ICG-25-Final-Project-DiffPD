import sys
sys.path.append('../')

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from py_diff_pd.common.common import create_folder, print_info
from py_diff_pd.env.armadillo_env_3d import ArmadilloEnv3d


def sweep_energy_vs_dt():
    # Prepare output folder
    folder = Path('armadillo_dt_sweep')
    create_folder(folder)

    # Initialize environment
    seed = 42
    env = ArmadilloEnv3d(seed, folder, {
        'youngs_modulus': 5e5,
        'init_rotate_angle': 0,
        'state_force_parameters': [0, 0, 0],  # No gravity
        'spp': 4
    })

    # Initial state
    deformable = env.deformable()
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = env.default_init_velocity()
    frame_num = 1  # single frame for energy measurement
    a0 = np.zeros(act_dofs)
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Time step sweep
    #dts = [1e-2, 5e-3, 2e-3, 1e-3, 1e-4]
    dts = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
    energies = []

    # Solver options for Newton
    method = 'newton_cholesky'
    opts = {
        'max_newton_iter': 5000,
        'max_ls_iter': 10,
        'abs_tol': 1e-9,
        'rel_tol': 1e-4,
        'verbose': 0,
        'thread_ct': 4
    }

    for dt in dts:
        print_info(f"Running dt = {dt:e}")
        loss, grad, info = env.simulate(
            dt, frame_num, method, opts,
            q0, v0, [a0], f0,
            require_grad=True,
            vis_folder=None
        )
        # loss here is final energy (elastic + PD)
        energies.append(loss)
        print_info(f"  Energy (loss) = {loss:.6f}")

    # Save raw data
    np.save(folder / 'dts.npy', np.array(dts))
    np.save(folder / 'energies.npy', np.array(energies))

    # Plot log–log curve
    # figure size
    plt.figure(figsize=(10, 6))
    plt.loglog(dts, energies, marker='o')
    plt.xlabel('Time step size (dt)')
    plt.ylabel('Loss (Elastic + PD Energy)')
    plt.title('Energy vs dt')
    plt.grid(True, which='both', linestyle='--')
    plt.savefig(folder / 'energy_vs_dt.png', dpi=300)
    plt.close()

    print_info(f"Results saved in {folder}")


if __name__ == '__main__':
    sweep_energy_vs_dt()
