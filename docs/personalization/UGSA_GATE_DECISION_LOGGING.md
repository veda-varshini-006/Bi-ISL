# UGSA Adaptation Decision Gate & Logging Spec (Prompt 45)

## Gating Formulation & Decision Function

The **UGSA Adaptation Decision Gate** (`UGSAGate`) determines whether live video stream predictions are sufficiently reliable to perform an online unsupervised adapter gradient step.

$$g_t = \mathbb{I}\left[ p_t \ge \tau_p \land q_t \ge \tau_q \land \Delta L_{\text{safe}} \le \epsilon \right]$$

---

## Validation-Derived Threshold Parameters

| Parameter | Symbol | Default Validation Value | Description |
| :--- | :---: | :---: | :--- |
| **Calibrated Confidence** | $\tau_p$ | `0.85` | Minimum temperature-calibrated sequence confidence |
| **Consensus Agreement** | $\tau_q$ | `0.75` | Minimum stochastic decoding sequence agreement rate |
| **Safety Degradation** | $\epsilon$ | `0.05` | Maximum allowable safety reference loss degradation ($5\%$) |

---

## JSONL Audit Telemetry Schema (`artifacts/logs/ugsa_gate_decisions.jsonl`)

Every adaptation decision step logs structured JSONL records:

```json
{
  "timestamp": "2026-09-01T23:35:00+00:00",
  "signer_id": "signer_102",
  "gate_decision": 1,
  "p_t": 0.8921,
  "q_t": 0.8333,
  "safety_delta": 0.0120,
  "tau_p": 0.85,
  "tau_q": 0.75,
  "epsilon": 0.05,
  "accept_reason": "CONFIDENCE_CONSENSUS_AND_SAFETY_VERIFIED",
  "reject_reason": ""
}
```
