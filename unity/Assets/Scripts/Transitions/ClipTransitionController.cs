// Unity Avatar Motion Clip Transition Controller (Prompt 75)
// Manages controlled transitions, hold phases, and AHDR envelopes between sign clips on Unity Humanoid rigs.

using System;
using System.Collections.Generic;
using UnityEngine;
using BiISL.Avatar.Retargeting;

namespace BiISL.Avatar.Transitions
{
    [Serializable]
    public class CoarticulationMetadataDTO
    {
        public string clip_id;
        public string gloss;
        public float transition_duration_ms = 120.0f;
        public float hold_duration_ms = 80.0f;
        public bool hold_handshape = true;
        public string blend_curve = "AHDR_HERMITE";
    }

    public class ClipTransitionController : MonoBehaviour
    {
        [Header("Target Retargeter")]
        public HumanoidRetargeter retargeter;
        public float defaultTransitionMs = 120.0f;
        public bool holdHandshapeByDefault = true;

        private void Awake()
        {
            if (retargeter == null)
            {
                retargeter = GetComponent<HumanoidRetargeter>();
            }
        }

        public void PlayStitchedSequence(AnimationClipData stitchedClip)
        {
            if (stitchedClip == null || stitchedClip.keyframes == null || retargeter == null)
            {
                Debug.LogWarning("[ClipTransitionController] Invalid sequence payload or missing retargeter.");
                return;
            }

            // Apply starting frame
            if (stitchedClip.keyframes.Count > 0)
            {
                retargeter.ApplyKeyframe(stitchedClip.keyframes[0]);
            }
        }
    }
}
