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
- ALL templates: ELHAM_RULES_K{2,3}
- Γ-invariant templates: ELHAM_SCALABLE_RULES_K{2,3}

### Execution policy (important)
We run the authors’ pipeline without removing `verify_log()`.
Therefore the reported "tmin" includes:
- minimization
- log verification
- explanation/statistics computation

This is documented explicitly when comparing against Table 4 and Table 5.

### Runs to execute

#### Table 4 (8 runs)
RF:
- ALL: k=2, k=3
- Γ-invariant: k=2, k=3

BPI-15:
- ALL: k=2, k=3
- Γ-invariant: k=2, k=3

#### Table 5 (8 runs)
RF:
- ALL: k=2, k=3
- Γ-invariant: k=2, k=3

BPI-15:
- ALL: k=2, k=3
- Γ-invariant: k=2, k=3

### Reporting policy
For each run, record:
- Table 4 fields: #inst, #sat, #min, tsat, tmin (as printed by the script)
- Table 5-related stats printed by `verify_log()` / explanation step
- Machine/OS/Python version and whether the run was repeated
