# Technical Avatar Specification: Bi-ISL 3D Rig & Procedural Animation Engine

**Document Version:** 1.0 (Phase 8 Specification)  
**Date:** September 2026  
**Status:** Approved / Specification Baseline  
**Traceability:** `O6`, `O8`, `RQ6`, `H5`, `E9`, `E10`, [`docs/RESEARCH_SPEC.md`](../RESEARCH_SPEC.md), [`docs/adr/0008-3d-avatar-representation.md`](../adr/0008-3d-avatar-representation.md)

---

## 1. Overview & Coordinate Architecture

This document defines the formal technical specification for the 3D avatar rendering subsystem within the Bi-ISL bidirectional communication framework. The avatar acts as the visual synthesis engine for reverse translation (English-to-ISL), converting structured ISL Intermediate Representations (IR) into linguistically faithful 3D skeletal motion keyframes and non-manual facial blendshapes.

### 1.1 Spatial Reference & Coordinate System
*   **Coordinate System:** Right-Handed Cartesian Coordinate System ($+X$ = Right, $+Y$ = Up, $+Z$ = Forward/Out of Screen).
*   **Rotational Order:** Euler $Z$-$X$-$Y$ intrinsic rotation sequence for skeletal joint transforms, converted internally to normalized quaternions $q = (w, x, y, z)$ where $\|q\| = 1$ to eliminate gimbal lock.
*   **Units of Measurement:**
    *   Spatial Positions: Meters ($\text{m}$) relative to avatar root origin at ground plane between ankles $(0, 0, 0)$.
    *   Rotations: Radians ($\text{rad}$) in API data contracts, degrees ($\text{deg}$) in configuration annotations.
    *   Time: Milliseconds ($\text{ms}$) normalized to an internal target frame rate of $60\text{ FPS}$ ($\Delta t = 16.67\text{ ms}$ per keyframe).

---

## 2. Humanoid Rig Topology & Bone Hierarchy

The avatar utilizes a standard humanoid skeletal rig featuring 67 total bones (excluding facial mesh attachment nodes), optimized for sign language articulation.

```
Root (Hips / Pelvis)
├── Spine_01 (Lumbar)
│   └── Spine_02 (Thoracic)
│       └── Spine_03 (Chest)
│           ├── Neck
│           │   └── Head
│           ├── Clavicle_L
│           │   └── UpperArm_L
│           │       └── LowerArm_L
│           │           └── Wrist_L
│           │               ├── Hand_L (Root of Left Hand)
│           │               │   ├── Thumb_01_L ── Thumb_02_L ── Thumb_03_L
│           │               │   ├── Index_01_L ── Index_02_L ── Index_03_L
│           │               │   ├── Middle_01_L ── Middle_02_L ── Middle_03_L
│           │               │   ├── Ring_01_L ── Ring_02_L ── Ring_03_L
│           │               │   └── Pinky_01_L ── Pinky_02_L ── Pinky_03_L
│           └── Clavicle_R
│               └── UpperArm_R
│                   └── LowerArm_R
│                       └── Wrist_R
│                           └── Hand_R (Root of Right Hand)
│                               ├── Thumb_01_R ── Thumb_02_R ── Thumb_03_R
│                               ├── Index_01_R ── Index_02_R ── Index_03_R
│                               ├── Middle_01_R ── Middle_02_R ── Middle_03_R
│                               ├── Ring_01_R ── Ring_02_R ── Ring_03_R
│                               └── Pinky_01_R ── Pinky_02_R ── Pinky_03_R
└── Leg_Root_L/R (Standard Lower Body Rig)
```

### 2.1 Bone Transform Specifications

| Bone Segment Name | Parent Node | Degrees of Freedom (DoF) | Primary Motion Range |
| :--- | :--- | :--- | :--- |
| `Hips` | `Root` | 6 DoF (3 Translation, 3 Rotation) | Base body placement & root displacement |
| `Spine_01` / `Spine_02` | `Hips` / `Spine_01` | 3 DoF (Rotation) | Torso sway, forward flex ($\pm 35^\circ$) |
| `Chest` (`Spine_03`) | `Spine_02` | 3 DoF (Rotation) | Upper torso rotation ($\pm 45^\circ$) & lean |
| `Neck` | `Chest` | 3 DoF (Rotation) | Head tilt ($\pm 30^\circ$), neck nod ($\pm 40^\circ$) |
| `Head` | `Neck` | 3 DoF (Rotation) | Fine head orientation & grammatical nod/shake |
| `Clavicle_L/R` | `Chest` | 2 DoF (Elevation, Protraction) | Shoulder shrug ($+30^\circ / -10^\circ$), roll |
| `UpperArm_L/R` | `Clavicle_L/R` | 3 DoF (Ball-and-Socket Rotation) | Shoulder abduction, flexion, rotation |
| `LowerArm_L/R` | `UpperArm_L/R` | 2 DoF (Flexion, Pronation/Supination) | Elbow bend ($0^\circ \text{ to } 145^\circ$), forearm twist |
| `Wrist_L/R` | `LowerArm_L/R` | 3 DoF (Flexion, Deviation, Twist) | Wrist pitch, yaw, roll articulation |

---

## 3. Independent Finger Joints

Hand configurations (handshapes) in ISL require precise, independent finger flexion, extension, abduction, and adduction. Both left and right hands feature 15 independent joints across 5 digits (30 total hand joints, 40 total finger DoFs).

```
Thumb:   [CMC / Trapeziometacarpal] ──> [MCP] ──> [IP]
Digits:  [MCP / Knuckle] ────────────> [PIP] ──> [DIP]
```

### 3.1 Anatomical Joint DoF & Range Mapping

| Digit | Joint Name | Node Name | DoF | Motion Types | Angular Limits (Flexion/Extension, Abduction) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Thumb** | CMC / Trapezium | `Thumb_01_L/R` | 2 | Flexion/Extension, Abduction | Flex: $0^\circ \text{ to } 60^\circ$, Abd: $0^\circ \text{ to } 70^\circ$ |
| | MCP | `Thumb_02_L/R` | 1 | Flexion/Extension | Flex: $0^\circ \text{ to } 55^\circ$, Ext: $0^\circ \text{ to } -10^\circ$ |
| | IP | `Thumb_03_L/R` | 1 | Flexion/Extension | Flex: $0^\circ \text{ to } 80^\circ$, Ext: $0^\circ \text{ to } -15^\circ$ |
| **Index** | MCP | `Index_01_L/R` | 2 | Flexion/Extension, Spread (Abd) | Flex: $0^\circ \text{ to } 90^\circ$, Abd: $-15^\circ \text{ to } +25^\circ$ |
| | PIP | `Index_02_L/R` | 1 | Flexion/Extension | Flex: $0^\circ \text{ to } 105^\circ$, Ext: $0^\circ$ |
| | DIP | `Index_03_L/R` | 1 | Flexion/Extension | Flex: $0^\circ \text{ to } 80^\circ$, Ext: $0^\circ$ |
| **Middle**| MCP | `Middle_01_L/R`| 2 | Flexion/Extension, Spread (Abd) | Flex: $0^\circ \text{ to } 90^\circ$, Abd: $-10^\circ \text{ to } +10^\circ$ |
| | PIP | `Middle_02_L/R`| 1 | Flexion/Extension | Flex: $0^\circ \text{ to } 105^\circ$, Ext: $0^\circ$ |
| | DIP | `Middle_03_L/R`| 1 | Flexion/Extension | Flex: $0^\circ \text{ to } 80^\circ$, Ext: $0^\circ$ |
| **Ring**  | MCP | `Ring_01_L/R`  | 2 | Flexion/Extension, Spread (Abd) | Flex: $0^\circ \text{ to } 90^\circ$, Abd: $-15^\circ \text{ to } +15^\circ$ |
| | PIP | `Ring_02_L/R`  | 1 | Flexion/Extension | Flex: $0^\circ \text{ to } 105^\circ$, Ext: $0^\circ$ |
| | DIP | `Ring_03_L/R`  | 1 | Flexion/Extension | Flex: $0^\circ \text{ to } 80^\circ$, Ext: $0^\circ$ |
| **Pinky** | MCP | `Pinky_01_L/R` | 2 | Flexion/Extension, Spread (Abd) | Flex: $0^\circ \text{ to } 90^\circ$, Abd: $-20^\circ \text{ to } +25^\circ$ |
| | PIP | `Pinky_02_L/R` | 1 | Flexion/Extension | Flex: $0^\circ \text{ to } 105^\circ$, Ext: $0^\circ$ |
| | DIP | `Pinky_03_L/R` | 1 | Flexion/Extension | Flex: $0^\circ \text{ to } 80^\circ$, Ext: $0^\circ$ |

### 3.2 Handshape Configuration Vectors
Each handshape (e.g., `FLAT_HAND_A`, `INDEX_POINT_1`, `FIST_S`, `FIVE_SPREAD`) is parameterized as a 20-dimensional normalized flex/abduction vector $\mathbf{h} \in [0, 1]^{20}$, mapped linearly to physical joint limits:
$$\theta_{j} = \theta_{j,\text{min}} + h_j \cdot (\theta_{j,\text{max}} - \theta_{j,\text{min}})$$

---

## 4. Wrist Articulation Subsystem

The wrist joint (`Wrist_L`, `Wrist_R`) serves as the terminal orientation anchor for manual signs. It mediates hand orientation (palm facing inward, outward, up, down, toward signer) and dynamic wrist wiggles/rotations.

### 4.1 Kinematic Channels & Limits
1.  **Wrist Flexion / Extension (Pitch):** Flexion $+75^\circ$ (toward inner forearm), Extension $-70^\circ$ (hand bent back).
2.  **Radial / Ulnar Deviation (Yaw):** Radial deviation $+20^\circ$ (thumb side bend), Ulnar deviation $-35^\circ$ (pinky side bend).
3.  **Forearm Pronation / Supination (Roll):** Supination $+90^\circ$ (palm up), Pronation $-90^\circ$ (palm down). *Note: Pronation-supination is shared between `LowerArm` roll and `Wrist` roll channels to prevent mesh distortion.*

### 4.2 Orientation Specification Matrix
Palm orientation vector $\mathbf{p} \in \mathbb{R}^3$ and finger direction vector $\mathbf{d} \in \mathbb{R}^3$ constrain wrist transform matrix $\mathbf{R}_{\text{wrist}} \in SO(3)$:
$$\mathbf{R}_{\text{wrist}} = \begin{bmatrix} \mathbf{d} \times \mathbf{p} & \mathbf{p} & \mathbf{d} \end{bmatrix}$$
Subject to orthonormalization via Gram-Schmidt process.

---

## 5. Elbow & Shoulder Motion Complex

Upper limb movement defines spatial signing trajectory (location in signing space: neutral space, chest level, face level, temple, side).

### 5.1 Kinematic Chain & Inverse Kinematics (IK)
The arm model implements a dual-mode **Forward Kinematics (FK)** and **Inverse Kinematics (IK)** solver:
*   **FK Mode:** Directly interpolates explicit joint angles $(\theta_{\text{shoulder\_flex}}, \theta_{\text{shoulder\_abd}}, \theta_{\text{shoulder\_rot}}, \theta_{\text{elbow\_flex}})$.
*   **Analytical 2-Bone IK Mode:** Solves upper arm ($L_1 = 0.28\text{ m}$) and lower arm ($L_2 = 0.26\text{ m}$) bone lengths to reach target wrist 3D coordinate $\mathbf{x}_{\text{target}} \in \mathbb{R}^3$ with elbow pole vector $\mathbf{x}_{\text{pole}}$ specifying elbow orientation out/down.

$$\cos(\theta_{\text{elbow}}) = \frac{\|\mathbf{x}_{\text{target}} - \mathbf{x}_{\text{shoulder}}\|^2 - L_1^2 - L_2^2}{2 L_1 L_2}$$

### 5.2 Motion Envelope & Self-Collision Boundaries
To prevent the avatar mesh from clipping into its own body during rapid chest/face signs, an elliptical capsule collision model bounds upper/lower arm IK target points:
*   **Torso Capsule Boundary:** Cylinder radius $R_{\text{torso}} = 0.18\text{ m}$, height $H = 0.55\text{ m}$.
*   **Head Sphere Boundary:** Sphere radius $R_{\text{head}} = 0.12\text{ m}$ centered at `Head` joint origin.

---

## 6. Head Movement & Non-Manual Grammatical Signals

Head movements encode essential grammatical functions in ISL (e.g., affirmations, negations, interrogative focus, topic boundary markers).

### 6.1 Rotational Channels & Grammatical Functions

| Channel | Axis | Motion Range | ISL Linguistic Function | Trigger Condition |
| :--- | :--- | :--- | :--- | :--- |
| **Nod (Pitch)** | $X$-axis | $-25^\circ \text{ (Down)} \text{ to } +20^\circ \text{ (Up)}$ | Affirmation / Topic Agreement | Statement assertion, confirmation |
| **Shake (Yaw)** | $Y$-axis | $-35^\circ \text{ (Left)} \text{ to } +35^\circ \text{ (Right)}$ | Negation (`NEG`) | Negative sentence intent (`INTENT_NEGATION`) |
| **Tilt (Roll)** | $Z$-axis | $-18^\circ \text{ (Left)} \text{ to } +18^\circ \text{ (Right)}$ | Question marking / Doubt (`WH-Q`, `YN-Q`) | Interrogative intent (`INTENT_QUESTION`) |
| **Forward Thrust**| $+Z$ Trans | $0 \text{ to } +0.06\text{ m}$ | Emphasis / Focus assertion | Intensive modifier / spatial contrast |

---

## 7. Facial Blendshapes & Non-Manual Marker Taxonomy

Facial expressions and mouthing are driven by a 32-target Facial Action Coding System (FACS) compatible blendshape dictionary. Each blendshape coefficient $w_k \in [0.0, 1.0]$.

### 7.1 Blendshape Target Dictionary

| Index | Blendshape Identifier | FACS Equivalent | ISL Grammatical Signal |
| :--- | :--- | :--- | :--- |
| 01 | `browOuterUpLeft` | AU 2L | Question / Surprise (`WH_QUESTION`) |
| 02 | `browOuterUpRight` | AU 2R | Question / Surprise (`WH_QUESTION`) |
| 03 | `browDownLeft` | AU 4L | Puzzlement / Furrowed brow (`YN_QUESTION`, `SERIOUS`) |
| 04 | `browDownRight` | AU 4R | Puzzlement / Furrowed brow (`YN_QUESTION`, `SERIOUS`) |
| 05 | `eyeSquintLeft` | AU 44L | Intensity / Focus / Small size |
| 06 | `eyeSquintRight` | AU 44R | Intensity / Focus / Small size |
| 07 | `eyeWideLeft` | AU 5L | Surprise / Large size (`BIG`, `AMAZED`) |
| 08 | `eyeWideRight` | AU 5R | Surprise / Large size (`BIG`, `AMAZED`) |
| 09 | `cheekPuff` | AU 33 | Large quantity / Heavy weight (`FAT`, `MANY`) |
| 10 | `cheekSuck` | AU 35 | Thinness / Small quantity (`THIN`, `SLIM`) |
| 11 | `jawOpen` | AU 26 | Open mouth / Vocalization / Exclamation |
| 12 | `mouthPucker` | AU 18 | Mouthing / Tight lip articulation (`SMALL`, `EXACT`) |
| 13 | `mouthFunnel` | AU 22 | Mouthing phoneme / Shape (`O_SHAPE`) |
| 14 | `mouthSmileLeft` | AU 12L | Pleasant / Affirmative affect |
| 15 | `mouthSmileRight` | AU 12R | Pleasant / Affirmative affect |
| 16 | `mouthFrownLeft` | AU 15L | Negative / Sad affect |
| 17 | `mouthFrownRight` | AU 15R | Negative / Sad affect |
| 18–32| `mouthing_phoneme_01..15`| Custom | Lexical ISL mouthings / English word visual cues |

---

## 8. Torso Movement & Spatial Locus Referencing

Torso movement provides spatial frame-of-reference shifts, role-shifting (direct discourse), and physical emphasis.

### 8.1 Torso Degrees of Freedom
1.  **Spine Flexion / Extension:** Torso forward lean ($0^\circ \text{ to } +20^\circ$) for emphasis; backward lean ($-10^\circ$) for hesitation/surprise.
2.  **Spine Lateral Bending:** Left/Right lean ($\pm 15^\circ$) for comparing two spatial entities (Locus A vs. Locus B).
3.  **Spine Yaw Rotation (Role Shift):** Torso turn left ($-25^\circ$) or right ($+25^\circ$) to adopt different character perspectives during conversational multi-turn dialogue.

---

## 9. Animation Timing, Envelopes & Coarticulation

To ensure natural animation playback without robotic jumps between discrete signs, motion curves are governed by parametric timing envelopes and smooth spline interpolation.

### 9.1 Attack-Hold-Decay-Release (AHDR) Envelope
Every manual sign motion sequence is divided into 4 temporal phases:

$$\text{Sign Duration } T_{\text{total}} = t_{\text{attack}} + t_{\text{hold}} + t_{\text{decay}} + t_{\text{release}}$$

```
Angle / Pos ^             +---------------+= Hold Phase (Peak Sign Geometry)
            |            /                 \
            |           /                   \
            |  Attack  /                     \ Decay / Release
            |  Phase  /                       \ Phase
            +--------+-------------------------+--------> Time (ms)
                    t0                       t1
```

*   **Attack Phase ($t_{\text{attack}} = 80 \text{--} 150\text{ ms}$):** Transition from rest pose or previous sign position into target sign locus.
*   **Hold Phase ($t_{\text{hold}} = 150 \text{--} 350\text{ ms}$):** Execution of primary manual sign geometry and movement stroke.
*   **Decay / Coarticulation Phase ($t_{\text{decay}} = 60 \text{--} 120\text{ ms}$):** Blending out toward neutral position or next sign entry envelope.
*   **Release Phase ($t_{\text{release}} = 100 \text{--} 200\text{ ms}$):** Return to neutral rest pose if sentence boundary is reached.

### 9.2 Coarticulation Spline Interpolation
Transitions between sign $K$ and sign $K+1$ use cubic Hermite / Catmull-Rom B-spline interpolation over bone quaternions via Spherical Linear Interpolation (Slerp):

$$\operatorname{Slerp}(q_1, q_2; u) = \frac{\sin((1-u)\Omega)}{\sin\Omega} q_1 + \frac{\sin(u\Omega)}{\sin\Omega} q_2$$

Where $\cos\Omega = q_1 \cdot q_2$, and $u \in [0, 1]$ is the normalized coarticulation blending parameter.

---

## 10. Left / Right Hand Independence & Dual-Arm Kinematics

ISL signs are categorized as one-handed (dominant hand only) or two-handed (dominant + non-dominant hands). The avatar architecture enforces strict kinematic separation between left and right hand pipelines.

### 10.1 Dominant vs. Non-Dominant Allocation Matrix

| Sign Category | Dominant Hand (`Hand_R` default) | Non-Dominant Hand (`Hand_L` default) | Coordination Rule |
| :--- | :--- | :--- | :--- |
| **One-Handed Sign** (e.g., `NAME`, `WHERE`) | Active trajectory & handshape | Rest pose at side or chest resting anchor | `Hand_L` pipeline disabled; zero velocity |
| **Two-Handed Symmetric** (e.g., `HELP`, `FAMILY`) | Active trajectory & handshape | Mirrored trajectory & handshape | Synchronized keyframes; $X$-axis reflected |
| **Two-Handed Asymmetric** (e.g., `WRITE`, `WORK`) | Active motion stroke | Base locus / stationary hand shape | `Hand_L` holds base shape; `Hand_R` signs on/near `Hand_L` |

### 10.2 Spatial Anchoring & Relational IK
For asymmetric signs where the dominant hand touches or moves relative to the non-dominant hand (e.g., placing dominant index finger onto non-dominant palm):
$$\mathbf{x}_{\text{target, R}} = \mathbf{T}_{\text{Wrist\_L}} \cdot \mathbf{x}_{\text{rel\_offset}}$$
Where $\mathbf{T}_{\text{Wrist\_L}}$ is the live transform matrix of the left wrist, guaranteeing relational accuracy regardless of torso or shoulder movement.

---

## 11. Conformance Checklist for Subsystem Verification

Any avatar renderer module, motion mapper, or blendshape engine in `src/avatar/` must comply with the following validation bounds:

- [x] **Bone Rig Hierarchy:** Must contain 67 standard bones matching exact naming in Section 2.
- [x] **Finger Independence:** Must expose 20 independent finger flexion/abduction DoF parameters per hand.
- [x] **Wrist Rotation:** Must execute 3 DoF rotational control without gimbal lock artifacts.
- [x] **Shoulder/Elbow IK:** Must prevent mesh self-collision via Section 5.2 capsule boundaries.
- [x] **Head Channels:** Must map `WH-Q` and `YN-Q` signals to neck pitch/yaw/roll channels.
- [x] **Blendshape Count:** Must include all 32 FACS/ISL blendshapes defined in Section 7.
- [x] **Torso Shift:** Must support 3 DoF spine shifts for dialogue role-shifting.
- [x] **Frame Rate Timing:** Keyframe timestamps normalized to $16.67\text{ ms}$ ($60\text{ FPS}$).
- [x] **Hand Independence:** Left and Right arms must be independently addressable with relational IK support.

---

**End of Technical Avatar Specification.**
