# process-mining-conformance-diagnostics-replication
Replication study of the BPM 2024 paper “Mining Behavioral Patterns for Conformance Diagnostics”, aiming to reproduce and analyze key evaluation results.
## External Artifacts

This replication uses the evaluation artifacts provided by the authors.

- Evaluation code and datasets:
  Zenodo DOI: https://doi.org/10.5281/zenodo.11480285

Due to size constraints, large artifacts are not stored in this repository.
Instructions below describe how to obtain and place them locally.

## Replication Scope

### Paper
Mining Behavioral Patterns for Conformance Diagnostics

### Goal
Replicate the paper’s evaluation on:
- Table 4: performance of constraint checking + minimization
- Table 5: diagnostic completeness/redundancy (deviant variants, explained moves, etc.)

### Datasets
- RF (road_fines_enriched)
- BPI-15 (BPI2015_1_8)

### Template sets and repository mapping
- ALL templates: ELHAM_RULES_K{2,3,3}
- Γ-invariant templates: ELHAM_SCALABLE_RULES_K{2,3,4}

### Execution policy (important)
We intentionally kept the diagnostic verification step enabled during all runs
This allowed us to measure end-to-end execution cost, not only discovery time
We analyzed how this affects runtime across:
- different k values
- ALL vs Γ-invariant templates

This highlights a practical trade-off between:
- fast constraint discovery
- full diagnostic execution
Therefore the reported "tmin" includes:
- minimization
- log verification
- explanation/statistics computation

This is documented explicitly when comparing against Table 4 and Table 5.

### How to run the replication experiments
This repository replicates the quantitative evaluation (Tables 4 and 5) from “Mining Behavioral Patterns for Conformance Diagnostics” using the authors’ released Python artifact.
#### 1) Prerequisites
- Windows / Linux / macOS
- Python 3.x installed (python --version)
- Enough disk space and RAM (k=4 runs can be heavy, we experimented with 16 GB RAM)

#### 2) Go to the original artifact folder
We keep the authors’ code unchanged under:
- `cd code/original/core_public`
#### 3) Create and activate a virtual environment
Windows (PowerShell):
- `python -m venv .venv`
- `.\.venv\Scripts\Activate.ps1`
#### 4) Install dependencies
- `pip install -r requirements.txt`

#### 5) Folder layout used by the artifact

The script expects these folders inside `code/original/core_public/`:
- `datasets/` (input datasets)
- `repositories/` (constraint template repositories)

To run one configuration at a time, we temporarily move all other files into `_hold/` so that the script loads only the dataset + repository we want.

We use:
- `datasets/_hold/` to store datasets not used in the current run
- `repositories/_hold/` to store repositories not used in the current run

#### 6) Select the dataset and repository for a run
Example: Road Fines + ALL templates + k=2

Inside `datasets/,` keep only:
- `road_fines_enriched.json`

Move all others into:
- `datasets/_hold/`

Inside `repositories/`, keep only:
- `ELHAM_RULES_K2.json`

Move all others into:
- `repositories/_hold/`

#### 7) Run the experiment and save the log
From `code/original/core_public/`, run:
Windows (PowerShell):
- `python core_public.py | Tee-Object -FilePath ..\..\..\results\logs\T4T5_RF_ALL_K2.log`

Repeat the same process for other configurations by swapping which single dataset and repository are left in place.

#### 8) Naming convention for logs
We save one log per configuration under `results/logs/`:

- T4T5_RF_ALL_K2.log
- T4T5_RF_ALL_K3.log
- T4T5_RF_ALL_K4.log
- T4T5_RF_SCALABLE_K2.log
- T4T5_RF_SCALABLE_K3.log
- T4T5_RF_SCALABLE_K4.log
- T4T5_BPI15_ALL_K2.log
- T4T5_BPI15_ALL_K3.log
- T4T5_BPI15_ALL_K4.log
- T4T5_BPI15_SCALABLE_K2.log
- T4T5_BPI15_SCALABLE_K3.log
- T4T5_BPI15_SCALABLE_K4.log

#### 9) Presentation
The final presentation used for the course is stored in:
`presentation/Replication_Presentation.pptx`