# Replication Plan

## Paper
Mining Behavioral Patterns for Conformance Diagnostics  
BPM 2024, Lecture Notes in Computer Science (LNCS 14940)

## Project Context
This work is part of a group project (5 members).  
Each member performs an initial individual replication before final consolidation.

## Replication Target
Reproduction of **Table 4 (Performance Evaluation)** from the paper.

The focus is on replicating the reported performance metrics for behavioral pattern
instantiation, satisfaction checking, and minimization.

## Dataset
**Road Fines** event log.

This dataset is selected because it uses a predefined reference model, which reduces
confounding factors related to process model discovery and allows a clearer evaluation
of the behavioral pattern mining pipeline.

## Experimental Scope
- Pattern instantiation up to:
  - `maxk = 2`
  - `maxk = 3`
- Evaluation metrics targeted (as reported in Table 4):
  - Number of instantiated patterns
  - Number of satisfied patterns
  - Number of minimal patterns
  - Runtime for satisfaction checking
  - Runtime for minimization

## Success Criteria
The replication is considered successful if:
- The evaluation pipeline can be executed end-to-end on the Road Fines dataset
- All Table 4 metrics are produced for `maxk = 2` and `maxk = 3`
- Deviations from the paper’s reported results are explicitly identified and explained

## Environment
(To be completed after setup)

- Operating System:
- Python Version:
- Hardware:
