# Senior-Thesis

Code and supporting analysis for my senior thesis on coarse-grained kinetic transition networks for LLPS-relevant hexapeptides.

This repository contains the code used to:
- build microscopic continuous-time Markov models from PATHSAMPLE / DPS outputs
- construct graph-transformed coarse models that preserve benchmark kinetics
- compute graph-theoretic descriptors on the reduced networks
- regress MFPTs and relaxation observables from those descriptors
- test node-feature committor baselines and sparse-message-passing GNN variants
- keep a small amount of exploratory work and compact generated summaries used during the project

## What Is Included

### Core KTN construction and I/O
- `config.py`
- `stationary_point_io.py`
- `io_markov.py`
- `build_markov_model.py`
- `build_gt_kept_models.py`
- `generate_basin_keep_lists.py`
- `run_all_build.py`

### Graph descriptors and kinetic analyses
- `graph_distances.py`
- `graph_features.py`
- `mfpt_analysis.py`
- `analyze_micro_vs_coarse_T300K.py`
- `make_micro_report.py`
- `landscape_class_tests.py`
- `summaries_and_regression.py`

### Classical ML for graph-level prediction
- `ml_regression.py`
- `ml_permutation_test.py`

### Committor baselines and GNN experiments
- `committor_linear_baseline.py`
- `ktn_dataset.py`
- `gnn_models.py`
- `train_gnn.py`
- `train_gnn_v2.py`
- `gnn_ablation_sweep.py`
- `gnn_ablation_aggregate.py`

### Cluster / wrapper scripts
- `*.sbatch`
- `*.slurm`
- `*.sh`

### Exploratory work and compact summaries
- `thesis_analysis.ipynb`
- `GTcheck_micro_vs_coarse_T300K_full.csv`
- `GTcheck_micro_vs_coarse_T300K_summary.txt`
- `micro_report_all.csv`
- `micro_report_all.txt`
- `gt_kept_build_report.txt`
- `CPU_PATCH_NOTES.txt`
- `qualitative_keeplist_checks.py`
- `quantitative_keeplist_checks.py`

## External Dependencies

Install the Python packages in `requirements.txt` first.

This code also depends on software that is not bundled in this repository:
- `PyGT` for KTN loading, graph transformation, and passage-time calculations
- Wales-group landscape tools such as `PATHSAMPLE`, `OPTIM`, and `GMIN`
- the underlying peptide stationary-point databases / DPS outputs

The raw stationary-point databases, PATHSAMPLE runs, and generated Markov-model folders are not included here.

## Environment Notes

The scripts were developed on a Princeton cluster workflow and some files still contain machine-specific paths, scratch directories, and SLURM settings.

Before running the pipeline on a new machine, update:
- `config.py`
- any `*.sbatch`, `*.slurm`, or `*.sh` launcher scripts you plan to use

## Typical Workflow

A representative end-to-end workflow is:
1. build microscopic KTNs with `build_markov_model.py`
2. construct graph-transformed coarse models with `build_gt_kept_models.py`
3. compute reduced-network descriptors with `graph_features.py`
4. summarize MFPT and relaxation behavior with `mfpt_analysis.py` and `analyze_micro_vs_coarse_T300K.py`
5. run graph-level prediction with `ml_regression.py` and `ml_permutation_test.py`
6. run committor baselines or GNN experiments with `committor_linear_baseline.py`, `train_gnn_v2.py`, and `gnn_ablation_sweep.py`

## Notes On Exploratory Files

Most files in this repository are part of the main supported pipeline. A few are exploratory scratch analyses kept here for completeness.

In particular, `quantitative_keeplist_checks.py` is exploratory and may require extra local utilities or unfinished helper code before it can run as-is. It is included for transparency, not as a polished entry point.

## What This Repo Does Not Contain

This repository is code only. It does not include the thesis LaTeX project, raw peptide databases, or the large generated model artifacts produced after PATHSAMPLE / PyGT processing.
