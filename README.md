## Recommended systems
- Ubuntu 18.04
- (Mini)conda 4.7.12 or higher
- GCC 7.5 (Other versions might work but we tested the codebase with 7.5 only)

## Installation
```
conda env create -f environment.yml
conda activate diff_pd
./install.sh
```

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
