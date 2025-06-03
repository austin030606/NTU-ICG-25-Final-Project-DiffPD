# NTU ICG 25 Final Project
## Disclaimer
The code included in the files in this GitHub repository are sourced from publicly available publications/code presented at DiffPD: Differentiable Projective Dynamics (ACM Transactions on Graphics/SIGGRAPH 2022). The project are used solely for academic communication and educational purposes. No commercial use is intended or involved.

## Recommended systems
- Ubuntu 18.04
- (Mini)conda 4.7.12 or higher
- GCC 7.5 (Other versions might work but we tested the codebase with 7.5 only)
- open3d=0.18.0 (conda install, pip might fail)

## Installation
```
conda env create -f environment.yml
conda activate diff_pd
./install.sh
```

If you are cloning the repository, make sure to specify --recursive

## Examples
Navigate to the `python/example` path and run `python [example_name].py` where the `example_name` could be the following names. By default, we use 8 threads in OpenMP to run PD simulation. This number can be modified in most of the scripts below by changing the `thread_ct` variable. It is recommended to set `thread_ct` to be **strictly smaller** than the number of cores available.

For more details about the python examples, you may navigate to the original DiffPD github repository at https://github.com/mit-gfx/diff_pd_public.

To run the examples in our report, you can run the `run_examples.sh` script inside `python/example`, or run the following commands individually:
```
python armadillo_3d.py > armadillo_3d_log_final.txt
python plant_3d_neo.py > plant_3d_neo_log_final.txt
python duck_3d_neo.py > duck_3d_neo_log_final.txt
```
The above commands may take a long time to run, so here is a [link](https://drive.google.com/file/d/1TVIK0p6gPkambhtDk_X_kSMunJpycAZO/view?usp=drive_link) to a zip file containing an example of the results 

To run the experiments in our reports:
```
# Exp:1 Loss vs. Time Steps
python armadillo_dt_sweep.py

# Exp:2 Tolerance (\texttt{rel\_tol} \& \texttt{abs\_tol}) vs Loss Drift \& Runtime
python armadillo_param_sweep_dt_pd_maxpd_tol.py

# Exp:3 Runtime vs Mesh Resolution
conda install -c conda-forge open3d=0.18.0
# Generate Two Intermediate Meshes
python generate_intermediate_meshes.py
# Runtime vs Mesh Resolution
python armadillop_mesh_resolution_vs_runtime_error.py
```

After running the Duck3D example, you can generate the corresponding figures using:
```
python python/example/print_duck_fig.py
```


## Files Added/Modified
Below is a list a files that we either modified from the original DiffPD source code or added ourselves:
- `cpp/core/include/fem/deformable.h`
- `cpp/core/include/solver/deformable_preconditioner.h`
- `cpp/core/src/fem/deformable.cpp`
- `cpp/core/src/fem/deformable_newton_forward.cpp`
- `cpp/core/src/fem/deformable_projective_dynamics_backward.cpp`
- `python/example/armadillo_dt_sweep.py`
- `python/example/armadillo_param_sweep_dt_pd_maxpd_tol.py`
- `python/example/armadillop_mesh_resolution_vs_runtime_error.py`
- `python/example/generate_intermediate_meshes.py`
- `asset/mesh/armadillo_mid1.obj`
- `asset/mesh/armadillo_mid2.obj`
- `python/example/armadillo_3d.py`
- `python/example/duck_3d_neo.py`
- `python/example/plant_3d_neo.py`
- `python/py_diff_pd/env/duck_env_3d_neo.py`
- `python/py_diff_pd/env/env_base.py`
- `python/py_diff_pd/env/plant_env_3d_neo.py`
- `python/example/print_duck_fig.py`

For more details, you may visit our github repo and browse our commit history at: https://github.com/austin030606/NTU-ICG-25-Final-Project-DiffPD

Our modifications consists of all the commits made after May 31, 2025.