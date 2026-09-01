# Naive Reverse Baseline Specification (Prompt 68)

## Overview

The **Naive Reverse Baseline** (`NaiveReverseBaseline`) serves as a simple, deterministic English-to-sign sequence translation baseline for Experiment **E9** benchmarking against our structured semantic-to-ISL generation pipeline.

---

## Baseline Characteristics vs Structured Pipeline

| Feature | Naive Lookup Baseline (`NaiveReverseBaseline`) | Structured Pipeline (`ISLPlanner`) |
| :--- | :--- | :--- |
| **Word Order** | English SVO (e.g. `ME HAVE A FEVER TODAY`) | ISL Time-SOV (e.g. `TODAY FEVER HAVE`) |
| **Non-Manual Markers** | ❌ None (`has_non_manual_markers: false`) | ✅ Facial & Eyebrow Annotations (`eyebrows_furrowed`, etc.) |
| **Semantic Abstraction** | ❌ None (Direct word lookup) | ✅ Domain Ontology (`ont:intent/*`, `ont:entity/*`) |
| **Spatial Loci** | ❌ None | ✅ Spatial Referents (`LOC_CENTER`, `LOC_RIGHT`) |
| **Co-articulation** | ❌ Hard cuts | ✅ Transition Hints (`smooth_blend`) |
