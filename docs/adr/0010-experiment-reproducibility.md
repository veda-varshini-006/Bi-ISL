# ADR-010: Experiment Reproducibility, Configuration & Artifact Management

**Status:** Accepted  
**Date:** September 2026  
**Deciders:** Bi-ISL Infrastructure & MLOps Team  
**Traceability:** `O1`, `O8`, `E0`–`E10`, [`docs/RESEARCH_SPEC.md`](../RESEARCH_SPEC.md#10-red-team-threat-model--failure-mitigation-strategy)

---

## Context

Scientific rigor requires bitwise exact experiment reproducibility. Deep learning literature routinely suffers from un-reproducible baselines, undocumented hyperparameter tweaks, hidden seed variations, data split leakage, and uncommitted script modifications.

For Bi-ISL, every experiment (E0–E10) must be 100% reproducible from declarative configuration files and split manifests without relying on manual setup steps.

---

## Decision

We decide to enforce **100% Declarative Configuration Management and Artifact Audit Protocols**:

1.  **Declarative YAML Configuration Files:** All dataset preprocessing options, model architectures, training hyperparameters, context attack settings, and evaluation metrics must be defined in version-controlled YAML files under `configs/`. Hardcoding settings inside Python scripts is strictly forbidden.
2.  **Explicit Split Manifests:** Dataset splits (train/dev/test) are locked via explicit JSON/CSV split manifests (`data/splits/`) containing video hashes, signer IDs, and clip boundaries to prevent data leakage (E0 audit).
3.  **Structured Experiment Logging:** Every experiment execution automatically logs random seeds, Git commit SHA, Python environment state, hardware specs, model checkpoints, and evaluation metrics into a structured JSON/YAML log artifact in `artifacts/`.
4.  **Seed Verification Grid:** Stochastic experiments (e.g., E6 joint mechanism grid) must be evaluated across at least 3 fixed random seeds, reporting mean, median, and 95% confidence intervals.

---

## Alternatives Considered

1.  **Command-Line Flag Passing (`argparse` / `sys.argv`):** Passing long command-line strings to scripts.  
    *Rejected:* Error-prone, hard to log, difficult to track complex nested configurations.
2.  **Hardcoded Python Configuration Constants (`config.py`):** Modifying Python files per run.  
    *Rejected:* Destroys git history, easily results in uncommitted state changes during experiment runs.

---

## Advantages

*   Guarantees exact bitwise experiment reproducibility across different compute environments.
*   Enables automated batch execution of the E0–E10 experimental program via CLI scripts (`src/experiments/`).
*   Simplifies paper submission audit trails and artifact evaluation reviews.

---

## Risks

*   Requires strict schema validation discipline to ensure configuration files do not drift over time.

---

## Consequences

*   `src/utils/config.py` implements Pydantic-based schema parsing and validation for all configuration files.
*   Experiment outputs are written to standardized directories under `artifacts/e0_leakage/`, `artifacts/e1_baseline/`, etc.

---

## Revisit Conditions

This ADR represents a core scientific engineering principle and is **Non-Revisitable**.
