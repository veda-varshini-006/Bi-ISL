# Machine-Actionable Research Specification: Bi-ISL

**Document Version:** 1.0 (Frozen Research Specification)  
**Date:** September 2026  
**Status:** Frozen / Approved for Baseline & Infrastructure Implementation  
**Source Proposals:** `BiISL_Proposal.pdf` (Bi-ISL-R Reconfigured Plan), `BiISL_First_Time_Proposal.pdf`, `BiISL_Execution_Playbook.pdf`  

---

## 1. Project Aim

To design and experimentally evaluate a context-gated, signer-adaptive, bidirectional Indian Sign Language (ISL) communication framework that translates continuous ISL into natural-language English output and converts spoken/text English into a structured ISL representation rendered through a 3D avatar, while preserving translation faithfulness, reliability under misleading context, and real-time feasibility under signer and environment variation.

---

## 2. Formal Research Objectives (O1–O8)

*   **O1 [Reproducible Baseline]:** Establish a reproducible continuous ISL-to-English baseline (SignVideo2Text) on public benchmarks (`ISLTranslate`, `iSign`, `INCLUDE`, `ISH-NEWS`) with standardized data split protocols.
*   **O2 [Shared Dialogue State]:** Design a Shared Bidirectional Dialogue State (SBDS) module storing compact structured conversational state $S_t = \{E_t, I_t, R_t, T_t, C_t, Y_{t-1}\}$ (entities, intents, referents, temporal/spatial attributes, confidence metadata, and previous turns).
*   **O3 [Context Reliability Gate]:** Develop a Context-Evidence Reliability Gate that dynamically estimates visual evidence vs. dialogue context reliability to suppress misleading, irrelevant, or hallucinated historical context.
*   **O4 [Uncertainty-Gated Personalization]:** Develop Uncertainty-Gated Signer Adaptation (UGSA) with predictive confidence checks, parameter update bounds, and protected-set rollback checks to prevent personalization drift and worst-signer performance degradation.
*   **O5 [Reverse Structured Generation]:** Develop a controlled-domain English-to-ISL structured generation pipeline translating English text/speech into an intermediate ISL representation (gloss sequence, temporal alignment, and non-manual marker specifications).
*   **O6 [Linguistically Grounded 3D Avatar]:** Map the intermediate ISL representation to a rigged 3D avatar animation engine for the reverse communication path, rendering manual signs and non-manual facial/head expressions.
*   **O7 [Mobile/Edge Deployment & Benchmarking]:** Optimize a lightweight model variant (quantization, operator fusion, pruning) for Android-class mobile hardware, benchmarking $p50/p95$ latency, peak RAM, storage footprint, and battery consumption.
*   **O8 [Systematic Validation & HCI Study]:** Execute controlled ablation experiments (E0–E9), adversarial context attacks, cross-signer generalization tests, and a Deaf and Hard-of-Hearing (DHH) user-centered avatar comprehension study (E10).

---

## 3. Research Questions (RQ1–RQ6)

*   **RQ1 [Gated Context Benefit]:** Does an evidence-gated conversational dialogue state (SBDS + Reliability Gate) improve continuous ISL-to-English translation quality compared with no-context and ungated context baselines on clean context-dependent benchmarks?
*   **RQ2 [Context Robustness & Misleading History]:** Does context reliability gating significantly reduce translation corruption and unsupported slot hallucinations when dialogue history is irrelevant, partially misleading, or explicitly contradictory?
*   **RQ3 [Safe Signer Personalization]:** Does Uncertainty-Gated Signer Adaptation (UGSA) improve per-signer translation metrics without degrading generalization on unseen signers or inducing catastrophic drift on protected baseline samples?
*   **RQ4 [Bidirectional Consistency]:** Does a Shared Bidirectional Dialogue State (SBDS) improve cross-turn entity/referent tracking and intent consistency across alternating forward (ISL-to-English) and reverse (English-to-ISL) dialogue turns compared with uncoupled single-turn translation?
*   **RQ5 [Mobile Real-Time Feasibility]:** Can a compressed variant of the bidirectional ISL system achieve sub-$200\text{ ms}$ $p95$ turn latency on target mid-range mobile hardware without exceeding acceptable translation quality degradation boundaries ($\le 1.5$ BLEU-4 drop)?
*   **RQ6 [Avatar Comprehension & Readability]:** Is the linguistically grounded structured 3D avatar presentation understandable and natural to ISL native/fluent users in controlled domain tasks compared to direct lookup baseline avatars and human signing references?

---

## 4. Testable Hypotheses (H1–H5)

*   **H1 [Context-Gated Translation Superiority]:** Evidence-gated context modeling (SBDS + Reliability Gate) will achieve statistically significant BLEU-4 and chrF++ improvements over a no-context baseline on context-dependent ISL translation datasets without causing a statistically significant increase in visually unsupported content (hallucinated slot rate $p > 0.05$).
*   **H2 [Resilience to Misleading Context]:** Under adversarial context perturbations (contradictory/irrelevant historical turns), the context-gated model will exhibit significantly lower semantic slot corruption and hallucinated slot rates than an ungated context baseline ($\Delta \text{Hallucination Rate} \ge 35\%$ reduction).
*   **H3 [Personalization Safety & Bounded Degradation]:** Uncertainty-Gated Signer Adaptation (UGSA) will increase average and median per-signer BLEU-4 / chrF++ scores while bounding worst-signer performance drop within a protected-set no-regression threshold ($\le 2.0\%$ drop relative to non-adapted base model).
*   **H4 [Cross-Turn Dialogue Coherence]:** Utilizing SBDS for bidirectional conversation will produce higher entity slot precision/recall across multi-turn interactions than uncoupled single-turn translation models.
*   **H5 [Mobile Real-Time Trade-off]:** An INT8/FP16 quantized model variant executing on target Android mobile hardware will maintain $p95$ end-to-end translation latency below $200\text{ ms}$ while preserving at least $95\%$ of the uncompressed model's BLEU-4 accuracy.

---

## 5. Scientific Novelty Boundaries & Prior-Work Contrast

To maintain academic rigor and prevent overclaiming, scientific contributions are bounded strictly against existing 2020–2026 literature:

| Prior Work / Benchmark | Capabilities Established | Explicit Boundary / What Bi-ISL Claims |
| :--- | :--- | :--- |
| **INCLUDE** (Sridhar et al., 2020) | 4,287 isolated sign videos; 263 signs | Used strictly for isolated pretraining/landmark extraction. *Isolated sign recognition is NOT claimed as novel.* |
| **ISLTranslate** (Joshi et al., ACL 2023) | ~31k continuous ISL-English sentence pairs | Standard sentence translation benchmark. *Dataset creation is NOT claimed as novel.* |
| **iSign** (Joshi et al., ACL 2024) | >118k ISL video-sentence pairs; multi-task benchmark | Benchmark dataset for Sign2Text and Text2Pose. *Multi-task benchmarking is NOT claimed as novel.* |
| **Contextual SLT** (Jang et al., CVPR 2025) | Adds previous turn transcripts and scene captions as context | Establishes that context helps SLT. *Bi-ISL claims evidence-gating to suppress misleading context.* |
| **SAME Signer Adaptation** (Yang et al., ACL 2026) | Signer-aware Mixture-of-Experts TTA for SLT | Establishes TTA for SLT. *Bi-ISL claims uncertainty gating with protected-set rollback to prevent drift.* |
| **Conversational Agent** (Nedungadi et al., SciRep 2025) | Real-time ISL context-aware agent for e-governance | Establishes context-aware ISL agent existence. *Bi-ISL claims shared bidirectional state & failure-bounded evaluation.* |
| **3D Dynamic Avatar** (IEEE RCSM 2025/2026) | Real-time procedural 3D avatar animation for signs | Establishes procedural avatar rendering. *Bi-ISL claims controlled semantic IR mapping with DHH comprehension validation.* |

### Primary Scientific Novelty Summary
1.  **Context-Evidence Reliability Gating:** A dynamic gating mechanism that evaluates spatiotemporal visual confidence against conversational context, preventing context-induced hallucination under misleading history.
2.  **Uncertainty-Gated Signer Adaptation (UGSA) with Protected Rollback:** An online adaptation strategy bounded by predictive uncertainty metrics and protected baseline verification to ensure non-degradation across diverse signers.
3.  **Shared Bidirectional Dialogue State (SBDS):** A unified lightweight representation ($S_t$) bridging forward continuous ISL video translation and reverse structured ISL avatar synthesis.
4.  **Empirical Red-Team Failure Analysis:** Rigorous evaluation under adversarial context attacks, pseudo-label noise injection, worst-signer degradation bounds, and edge device thermal/latency constraints.

---

## 6. Experimental Program (E0–E10 Matrix)

| Exp ID | Experiment Title | Objective & Description | Baselines Compared | Primary Evaluation Metric |
| :--- | :--- | :--- | :--- | :--- |
| **E0** | **Data & Split Leakage Audit** | Verify dataset integrity, audit train/dev/test splits, and enforce zero overlap in video sources or signers. | Official vs Constructed Splits | Overlap ratio ($= 0.0\%$), Duplicate frame count |
| **E1** | **Baseline SLT Reproduction** | Reproduce defensible continuous ISL translation baselines on standard benchmarks. | B0 (Visual-only Transformer/BiLSTM) | BLEU-4, chrF++, ROUGE-L |
| **E2** | **Context Ablation** | Evaluate forward SLT accuracy under clean, ground-truth context across dialogue turns. | B0 (No context), B1 (Raw transcript history), B2 (Ungated SBDS), M1 (Gated SBDS) | BLEU-4, BERTScore, Context-subset BLEU |
| **E3** | **Misleading-Context Attack** | Stress-test context gating by injecting irrelevant, partially misleading, and contradictory history turns. | B1 (Raw context), B2 (Ungated SBDS), M1 (Gated SBDS) | Unsupported Slot Rate (Hallucinations), Semantic Slot Precision |
| **E4** | **Signer Personalization Test** | Evaluate online/test-time adaptation across individual signers. | B0 (Generic base), Supervised Upper Bound, P1 (Naive TTA), SAME baseline, P2 (UGSA) | Per-signer $\Delta$ BLEU-4, Median Signer Gain, Worst-Signer Drop |
| **E5** | **Adaptation Noise & Drift Test** | Inject noisy pseudo-labels / erroneous feedback to evaluate adaptation safety and rollback logic. | P1 (Unsafe TTA), P2 (UGSA with Rollback) | Protected-set regression rate, Rollback trigger frequency, Recovery turns |
| **E6** | **Joint Mechanism Interaction** | Measure interaction grid between context gating and signer adaptation across random seeds. | Base, SBDS-only, UGSA-only, SBDS + UGSA (FULL) | Joint BLEU-4, chrF++, ECE (Calibration) |
| **E7** | **Cross-Signer & Shift Generalization** | Test model robustness against unseen signers, background/lighting shifts, and cross-domain corpora. | Full Model on Clean vs Shifted Sets | Out-of-domain BLEU-4 degradation %, GER |
| **E8** | **Mobile Edge Benchmark** | Measure real-time execution performance on target Android mobile devices across precision formats. | FP32 Base, FP16 Quantized, INT8 Quantized on target phone | $p50/p95$ Latency ($\text{ms}$), Peak RAM ($\text{MB}$), Storage ($\text{MB}$), Battery proxy |
| **E9** | **Reverse Generation & Avatar Test** | Evaluate accuracy of English-to-ISL structured representation and sign motion mapping. | Direct Word-Gloss Lookup vs Controlled Semantic IR Pipeline | Gloss Alignment Accuracy, Non-Manual Marker Precision |
| **E10**| **DHH User Comprehension Study** | Conduct within-subject user study with Deaf and Hard-of-Hearing (DHH) participants evaluating avatar output. | C1 (Human Video Ref), C2 (Legacy Avatar), C3 (Proposed Avatar) | Comprehension Accuracy (%), Task Success (%), SUS Usability Score |

---

## 7. Metrics & Evaluation Standards

### 7.1 Primary Metrics
*   **BLEU-4 (Bilingual Evaluation Understudy):** Primary surface n-gram translation precision metric for continuous ISL-to-English translation.
*   **chrF++:** Character n-gram F-score with word bi-grams; primary translation quality metric tolerant of morphological variation.
*   **BERTScore:** Contextual embedding semantic similarity measuring meaning preservation beyond exact surface text overlap.
*   **Unsupported Slot Rate (USR):** Primary context reliability/faithfulness metric measuring the proportion of generated entity/attribute slots not backed by visual evidence (evaluates context hallucination during E3 attacks).
*   **Per-Signer $\Delta$ BLEU-4 & Worst-Signer Degradation ($\Delta \text{BLEU}_{\text{worst}}$):** Primary personalization safety metrics measuring individual signer gains and ensuring worst-signer performance drop is bounded ($\le 2.0\%$).
*   **Expected Calibration Error (ECE):** Primary confidence calibration metric evaluating predictive probability reliability for UGSA update thresholding.
*   **$p50$ and $p95$ End-to-End Latency ($\text{ms}$):** Primary mobile deployment responsiveness metric measured on target Android hardware ($p95 < 200\text{ ms}$).
*   **Objective Avatar Comprehension Accuracy (%):** Primary HCI metric measuring percentage of correct responses by DHH participants in controlled domain tasks.

### 7.2 Secondary Metrics
*   **Gloss Error Rate (GER) / Word Error Rate (WER):** Secondary alignment/recognition metric evaluated only where gloss ground-truth annotations exist.
*   **ROUGE-L:** Secondary surface sequence overlap metric evaluating longest common subsequence match.
*   **Semantic Slot Precision & Recall:** Secondary entity extraction metrics measuring turn-level dialogue attribute extraction accuracy.
*   **Frames Per Second (FPS):** Secondary throughput metric evaluating visual feature extraction and temporal decoding speed.
*   **Peak RAM ($\text{MB}$) & Storage Footprint ($\text{MB}$):** Secondary resource usage metrics on target deployment hardware.
*   **Energy / Battery Proxy ($\% \Delta/\text{hr}$) & Thermal Throttling Drop (%):** Secondary power and thermal stability metrics measured over repeated continuous execution.
*   **Task Completion Rate (%) & System Usability Scale (SUS):** Secondary subjective and interaction usability metrics (0–100 scale).

---

## 8. Datasets & Data Protocols

1.  **ISLTranslate** (~31,000 sentence pairs): Primary continuous ISL-to-English translation training and dev benchmark.
2.  **iSign** (>118,000 video-text/pose pairs): Large-scale continuous ISL processing benchmark used for pretraining and multi-turn dialog evaluation.
3.  **INCLUDE** (4,287 isolated videos, 263 signs): Used exclusively for isolated sign feature extractor pretraining and landmark normalization.
4.  **ISH-NEWS**: Continuous ISL news domain dataset used for external cross-domain generalization testing (E7).
5.  **E3 Context Stress-Test Corpus:** Synthetic/curated dialogue pairs generated with explicit context perturbations:
    *   *Irrelevant Context:* History from an unrelated domain.
    *   *Partially Misleading Context:* Overlapping entity names with altered actions/attributes.
    *   *Contradictory Context:* Direct negation of present visual facts in dialogue history.

---

## 9. Baseline Definitions

*   **B0 [No-Context Visual Baseline]:** Spatiotemporal visual encoder + Transformer sequence decoder operating solely on current video frames without historical state.
*   **B1 [Naive Transcript Context Baseline]:** Appends raw text transcript of previous turn $Y_{t-1}$ directly to decoder input sequence (Jang et al. style).
*   **B2 [Ungated Dialogue State Baseline]:** Incorporates compact dialogue state $S_t$ into decoder attention without context reliability gating.
*   **P1 [Naive Unsafe Adaptation]:** Applies online fine-tuning on pseudo-labeled target signer data without confidence thresholding or rollback checks.
*   **SAME Baseline:** Modern Signer-Aware Mixture-of-Experts test-time adaptation baseline (Yang et al., ACL 2026).
*   **C1 [Human Reference Video]:** Original recorded video of native/fluent ISL signer (Upper bound reference).
*   **C2 [Legacy Direct Lookup Avatar]:** Rule-based direct word-to-sign dictionary lookup rendering avatar without intermediate linguistic representation or non-manual markers.

---

## 10. Red-Team Threat Model & Failure Mitigation Strategy

| Failure Threat / Stress Test | Red-Team Attack Protocol | Expected Safe System Behavior | Failure Exit Action |
| :--- | :--- | :--- | :--- |
| **Misleading Context Injection** | Inject contradictory prior turn text into context window during E3. | Reliability gate down-weights context; output relies strictly on visual features. | Abstain or output low-confidence warning if visual signal is ambiguous. |
| **Adaptation Poisoning / Drift** | Supply incorrect pseudo-labels or noisy sign inputs during online adaptation. | Uncertainty gate blocks gradient update; protected-set check triggers rollback. | Automatic rollback to last verified checkpoint $W_{\text{safe}}$. |
| **Severe Signer / Environment Shift** | Feed un-seen signer in extreme low-light / occluded setting. | Predictive uncertainty increases; output confidence is properly calibrated. | Graceful degradation without confident hallucinations. |
| **Avatar Representation Gap** | Request translation for text containing unmapped / missing ISL signs. | Generation pipeline detects missing sign asset; avoids inventing unvalidated movement. | Fall back to finger-spelling or text subtitle overlay. |
| **Mobile Resource Throttling** | Execute repeated inference under simulated thermal throttling & memory pressure. | System drops optional non-manual feature passes or lowers frame sampling rate. | Preserve execution stability; avoid crashes or indefinite freezes. |

---

## 11. Publication Strategy & Claim Discipline

### 11.1 Defensible Allowed Claims (Must be backed by E0–E10 empirical evidence)
*   ✅ *"Under the specified benchmarks (ISLTranslate, iSign), the proposed evidence-gated shared dialogue state improves context-dependent continuous ISL translation relative to no-context and naive-context baselines while reducing hallucinated slot errors under misleading-history perturbations."*
*   ✅ *"Uncertainty-Gated Signer Adaptation (UGSA) improves median target-signer translation metrics while restricting worst-signer performance drop within a protected no-regression threshold via automatic rollback."*
*   ✅ *"The optimized mobile model variant achieves sub-200ms $p95$ end-to-end latency on mid-range Android hardware with acceptable translation quality trade-offs."*

### 11.2 Explicitly Forbidden Claims (Strictly Prohibited)
*   ❌ *"First context-aware Indian Sign Language translation system."*
*   ❌ *"First bidirectional Indian Sign Language communication system."*
*   ❌ *"First personalized sign language translator."*
*   ❌ *"First 3D avatar English-to-ISL communication tool."*
*   ❌ *"Context always improves translation accuracy."*
*   ❌ *"Personalization improves all signers without any degradation."*
*   ❌ *"Solves open-domain English-to-ISL translation."*

---

## 12. System Traceability Matrix

The following matrix maps every research requirement (Aim, Objectives O1–O8, Research Questions RQ1–RQ6, Hypotheses H1–H5) directly to its implementation module, evaluation experiment, primary metric, and target paper section.

| Requirement ID | Requirement Summary | Implementation Module | Experiment | Primary Evaluation Metric | Target Paper Section |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **O1 / RQ1 / H1** | Reproducible baseline & context-gated SLT superiority | `src/models/baseline_slt.py`<br>`src/models/context_gate.py` | E1, E2 | BLEU-4, chrF++, BERTScore, Context BLEU | Sec 4.1 Continuous SLT & Context Gating |
| **O2 / RQ4 / H4** | Shared Bidirectional Dialogue State (SBDS) entity & intent tracking | `src/dialogue/sbds_manager.py` | E2, E6 | Entity Slot Precision/Recall, Intent Accuracy | Sec 4.2 Shared Dialogue State |
| **O3 / RQ2 / H2** | Context reliability gating under misleading context attacks | `src/models/reliability_gate.py` | E3 | Unsupported Slot Rate (USR), Semantic Corruption % | Sec 4.3 Context Reliability & Stress Attacks |
| **O4 / RQ3 / H3** | Uncertainty-gated signer adaptation (UGSA) & safe rollback | `src/adaptation/ugsa_adapter.py`<br>`src/adaptation/rollback.py` | E4, E5 | Per-signer $\Delta$ BLEU-4, Worst-Signer Drop, Rollback Freq | Sec 4.4 Personalization & Safety Rollback |
| **O5 / RQ6** | Reverse English-to-ISL structured representation generation | `src/reverse/structured_gen.py` | E9 | Gloss Sequence Accuracy, Non-Manual Marker Precision | Sec 5.1 Reverse ISL Generation Pipeline |
| **O6 / RQ6 / H5** | Linguistically grounded 3D avatar motion rendering | `src/avatar/avatar_renderer.py` | E9, E10 | Avatar Comprehension Accuracy (%), SUS Score | Sec 5.2 3D Avatar Rendering & HCI |
| **O7 / RQ5 / H5** | On-device mobile optimization & real-time benchmarking | `src/deployment/mobile_export.py` | E8 | $p50/p95$ Latency ($\text{ms}$), Peak RAM ($\text{MB}$), Footprint | Sec 6.1 Mobile Edge Benchmarking |
| **O8 / RQ1-RQ6** | Full systematic ablation grid & DHH user validation | `src/eval/ablation_runner.py` | E0–E10 | Complete Metric Suite (E0–E10) | Sec 7 Discussion, Ablations & HCI Study |

---

**End of Research Specification.**  
*This specification is frozen. No implementation of deep learning models or code modules should proceed without strict adherence to the parameters, metrics, and novelty boundaries defined herein.*
