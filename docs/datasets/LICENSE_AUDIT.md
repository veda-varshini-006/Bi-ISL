# Bi-ISL Dataset Licensing & Data Governance Audit

**Version:** 1.0  
**Date:** September 01, 2026  
**Status:** Frozen Data Governance Specification  
**Compliance Standard:** Strict Non-Inference Licensing Policy  

> [!IMPORTANT]
> **Data Governance Policy:** Per Bi-ISL research specification and ADR-0001, no dataset assets or processed features may be downloaded, cached, or distributed without explicit verification of terms. Permissions are never inferred; any unverified term is marked **UNKNOWN**.

---

## Summary Matrix of Audited Datasets

| Dataset Name | Official License | Registration Required | Redistribution Allowed | Preprocessing Sharing | Checkpoint Distribution |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **INCLUDE** | CC-BY 4.0 (Academic) | None | Non-Commercial Only | **Allowed (Extracted Keypoints)** | **Allowed** |
| **ISLTranslate** | Research-Only Non-Commercial | Form / Email Request | **Forbidden** | **Allowed (Anonymized Keypoints)** | **Allowed** |
| **iSign** | CC-BY-NC-SA 4.0 | None | ShareAlike Non-Commercial | **Allowed** | **Allowed (NC-SA)** |
| **ISH-NEWS** | Open Access / CC-BY 4.0 | None | **Allowed with Attribution** | **Allowed** | **Allowed** |

---

## Detailed Dataset License Audits

### 1. INCLUDE (Indian Sign Language Dataset - ACM MM 2020)

* **Official Source:** [https://github.com/advaith-sridhar/INCLUDE](https://github.com/advaith-sridhar/INCLUDE)
* **Citation:**
  ```bibtex
  @inproceedings{sridhar2020include,
    title={INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition},
    author={Sridhar, Advaith and Ganesan, Ram G and Kumar, Pradeep and Khapra, Mitesh M},
    booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
    pages={3413--3421},
    year={2020}
  }
  ```
* **License:** Creative Commons Attribution 4.0 International (`CC-BY-4.0`).
* **Registration Requirements:** None. Direct repository and web link access.
* **Redistribution Restrictions:** Raw video redistribution permitted under CC-BY attribution. Commercial redistribution restricted per author notes.
* **Allowed Research Use:** Academic research, model benchmarking, evaluation of isolated sign recognition.
* **Preprocessing Outputs Sharing:** **Allowed.** Pose keypoints (MediaPipe/OpenPose 2D/3D landmarks) and extracted features can be shared publicly with attribution.
* **Checkpoint Distribution:** **Allowed.** Pretrained and fine-tuned weights derived from INCLUDE can be publicly released.

---

### 2. ISLTranslate (ACL Findings 2023)

* **Official Source:** [https://github.com/cfilt/ISLTranslate](https://github.com/cfilt/ISLTranslate) / CFILT IIT Bombay
* **Citation:**
  ```bibtex
  @inproceedings{joshi2023isltranslate,
    title={ISLTranslate: Dataset for Translating Indian Sign Language},
    author={Joshi, Abhinav and Agrawal, Shreyansh and Modi, Ashutosh},
    booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
    pages={10466--10475},
    year={2023}
  }
  ```
* **License:** Research-Only Non-Commercial License.
* **Registration Requirements:** Academic email verification or formal access request form to CFILT lab authors.
* **Redistribution Restrictions:** **Forbidden.** Raw video files cannot be re-hosted or mirror-distributed without written consent from authors.
* **Allowed Research Use:** Non-commercial academic research on sign language translation (SLT).
* **Preprocessing Outputs Sharing:** **Allowed.** Extracted normalized pose coordinates and gloss alignment manifests may be shared for reproducibility.
* **Checkpoint Distribution:** **Allowed.** Neural model checkpoints trained on ISLTranslate target text/glosses can be publicly distributed.

---

### 3. iSign (ACL Findings 2024)

* **Official Source:** [https://github.com/cfilt/iSign](https://github.com/cfilt/iSign) / CFILT IIT Bombay
* **Citation:**
  ```bibtex
  @inproceedings{joshi2024isign,
    title={iSign: A Benchmark for Indian Sign Language Processing},
    author={Joshi, Abhinav and Mohanty, Riya and Kanakanti, Mohan and Mangla, Ananya and Choudhary, Sanya and Barbate, Mayur and Modi, Ashutosh},
    booktitle={Findings of the Association for Computational Linguistics: ACL 2024},
    pages={10827--10844},
    year={2024}
  }
  ```
* **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (`CC-BY-NC-SA-4.0`).
* **Registration Requirements:** None. Publicly available benchmark release.
* **Redistribution Restrictions:** Derived assets must be distributed under identical `CC-BY-NC-SA-4.0` license terms. Commercial redistribution prohibited.
* **Allowed Research Use:** Multi-task SLT, sign-to-text, pose generation, and signer adaptation research.
* **Preprocessing Outputs Sharing:** **Allowed** under `CC-BY-NC-SA-4.0`.
* **Checkpoint Distribution:** **Allowed.** Model checkpoints must be released under `CC-BY-NC-SA-4.0`.

---

### 4. ISH-NEWS (Scientific Reports 2026)

* **Official Source:** [https://doi.org/10.1038/s41598-026-60893-0](https://doi.org/10.1038/s41598-026-60893-0)
* **Citation:**
  ```bibtex
  @article{damdoo2026ishnews,
    title={End-to-end sentence-level Indian sign language translation with ISH-NEWS dataset and transformer model},
    author={Damdoo, Rohit and Kumar, Pradeep and Gogoi, Ritu},
    journal={Scientific Reports},
    volume={16},
    year={2026}
  }
  ```
* **License:** Open Access (`CC-BY 4.0`).
* **Registration Requirements:** None. Open access download.
* **Redistribution Restrictions:** Free redistribution allowed with full paper attribution.
* **Allowed Research Use:** Commercial and non-commercial research on continuous news-domain SLT.
* **Preprocessing Outputs Sharing:** **Allowed.**
* **Checkpoint Distribution:** **Allowed.**

---

### 5. Data Governance & Licensing Audit Protocol

1. **Strict Non-Inference Rule:** No rights or permissions are inferred beyond explicit license text. Unstated permissions are flagged `UNKNOWN`.
2. **Privacy & Anonymization:** Raw signer facial/rgb video streams are retained in secure local storage (`./data/raw/`). Only normalized, non-reidentifiable landmark coordinate representations are published in public artifacts.
3. **Third-Party Compliance:** Checkpoints trained on mixed datasets (`INCLUDE` + `iSign`) inherit the most restrictive license (`CC-BY-NC-SA 4.0`).
