# DML Hyperparameter Tuning Replication Files: Motivation Example

This directory contains the replication code for the motivating example Figures 1 (Section), based on the data BCH generating process (Belloni et al., 2013).

The directory is organized as follows

- `code` - contains the source code for the simulation study, including the data generating process
- `simulation_run` - contains the scripts to run the simulation study, organized in the following subdirectories
    - `sim_lassomanual.py` - Code to run the simulation study in various setting; the setting considered in the motivation example corresponds to the parameters `this_design = "1a"`, `rho = 0.6` and `R2 = 0.6`
    - `results` - results as `.csv` files
- An evaluation and replication of the surface plots in the paper is available [here](). **TODO**: Add link to notebook.


## References

Alexandre Belloni, Victor Chernozhukov, and Christian Hansen. Inference on Treatment Effects after Selection among High-Dimensional Controls. *The Review of Economic Studies*, 81(2):608–650, 11 2013. ISSN 0034-6527. doi: 10.1093/restud/rdt044. URL https://doi.org/10.1093/restud/rdt044.