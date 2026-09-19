// Unity Avatar Non-Manual Marker (NMM) Controller (Prompt 76)
// Drives facial blendshapes, Head bone rotations, and Spine body lean transforms on Unity Humanoid rigs.

using System;
using System.Collections.Generic;
using UnityEngine;

namespace BiISL.Avatar.NMM
{
    [Serializable]
    public class NonManualTagPayload
    {
        public string tag;
        public string category;
        public float start_ms;
        public float end_ms;
        public float intensity = 1.0f;
    }

    [Serializable]
    public class NMMFrameStatePayload
    {
        public float timestamp_ms;
        public List<string> active_tags;
        public string eyebrow_state;
        public Dictionary<string, float> blendshapes;
        public float[] head_pose_euler; // [pitch, yaw, roll]
        public float[] body_lean_euler; // [pitch, yaw, roll]
    }

    public class NMMController : MonoBehaviour
    {
        [Header("Target Rendering References")]
        public SkinnedMeshRenderer faceMeshRenderer;
        public Animator targetAnimator;

        [Header("Runtime State")]
        public string currentEyebrowState = "NEUTRAL";
        public List<string> currentActiveTags = new List<string>();

        private Transform headBone;
        private Transform spineBone;

        private void Awake()
        {
            if (targetAnimator == null)
            {
                targetAnimator = GetComponent<Animator>();
            }

            if (targetAnimator != null && targetAnimator.isHuman)
            {
                headBone = targetAnimator.GetBoneTransform(HumanBodyBones.Head);
                spineBone = targetAnimator.GetBoneTransform(HumanBodyBones.Spine);
            }
        }

        public void ApplyNMMFrameState(NMMFrameStatePayload payload)
        {
            if (payload == null) return;

            currentEyebrowState = payload.eyebrow_state ?? "NEUTRAL";
            currentActiveTags = payload.active_tags ?? new List<string>();

            // 1. Apply Blendshapes on SkinnedMeshRenderer
            if (faceMeshRenderer != null && faceMeshRenderer.sharedMesh != null && payload.blendshapes != null)
            {
                Mesh mesh = faceMeshRenderer.sharedMesh;
                foreach (var kvp in payload.blendshapes)
                {
                    int index = mesh.GetBlendShapeIndex(kvp.Key);
                    if (index >= 0)
                    {
                        // Convert [0.0, 1.0] weight to Unity [0, 100] blendshape weight
                        faceMeshRenderer.SetBlendShapeWeight(index, Mathf.Clamp01(kvp.Value) * 100.0f);
                    }
                }
            }

            // 2. Apply Head Pose Rotations
            if (headBone != null && payload.head_pose_euler != null && payload.head_pose_euler.Length >= 3)
            {
                float pitchDeg = payload.head_pose_euler[0] * Mathf.Rad2Deg;
                float yawDeg = payload.head_pose_euler[1] * Mathf.Rad2Deg;
                float rollDeg = payload.head_pose_euler[2] * Mathf.Rad2Deg;

                headBone.localRotation = Quaternion.Euler(pitchDeg, yawDeg, rollDeg);
            }

            // 3. Apply Body Lean Rotations on Spine
            if (spineBone != null && payload.body_lean_euler != null && payload.body_lean_euler.Length >= 3)
            {
                float pitchDeg = payload.body_lean_euler[0] * Mathf.Rad2Deg;
                float yawDeg = payload.body_lean_euler[1] * Mathf.Rad2Deg;
                float rollDeg = payload.body_lean_euler[2] * Mathf.Rad2Deg;

                spineBone.localRotation = Quaternion.Euler(pitchDeg, yawDeg, rollDeg);
            }
        }
    }
}
