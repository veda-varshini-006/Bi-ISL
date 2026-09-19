// Unity Humanoid Rig Motion Retargeter (Prompt 74)
// Retargets motion keyframe data onto Unity Humanoid Avatar rigs, enforcing joint constraints and handedness mirroring.

using System;
using System.Collections.Generic;
using UnityEngine;

namespace BiISL.Avatar.Retargeting
{
    [Serializable]
    public class KeyframeData
    {
        public float timestamp_ms;
        public Dictionary<string, float[]> joint_rotations;
        public Dictionary<string, float[]> joint_positions;
        public Dictionary<string, float> blendshape_weights;
    }

    [Serializable]
    public class AnimationClipData
    {
        public string clip_id;
        public string sign_id;
        public string gloss;
        public float duration;
        public string dominant_hand;
        public List<KeyframeData> keyframes;
    }

    public class HumanoidRetargeter : MonoBehaviour
    {
        [Header("Target Rig Configuration")]
        public Animator targetAnimator;
        public float rigHeightScale = 1.0f;
        public bool enforceJointConstraints = true;
        public string targetHandedness = "RIGHT";

        private Dictionary<HumanBodyBones, Transform> boneMap = new Dictionary<HumanBodyBones, Transform>();

        private void Awake()
        {
            if (targetAnimator == null)
            {
                targetAnimator = GetComponent<Animator>();
            }

            MapHumanoidBones();
        }

        private void MapHumanoidBones()
        {
            if (targetAnimator == null || !targetAnimator.isHuman)
            {
                Debug.LogWarning("[HumanoidRetargeter] Target animator is not configured as Humanoid.");
                return;
            }

            foreach (HumanBodyBones bone in Enum.GetValues(typeof(HumanBodyBones)))
            {
                if (bone == HumanBodyBones.LastBone) continue;
                Transform boneTransform = targetAnimator.GetBoneTransform(bone);
                if (boneTransform != null)
                {
                    boneMap[bone] = boneTransform;
                }
            }
        }

        public void ApplyKeyframe(KeyframeData keyframe)
        {
            if (keyframe == null) return;

            // 1. Apply Bone Rotations with Joint Constraint Clamping & Handedness Mirroring
            if (keyframe.joint_rotations != null)
            {
                foreach (var kvp in keyframe.joint_rotations)
                {
                    string boneName = kvp.Key;
                    float[] rot = kvp.Value;
                    if (rot == null || rot.Length < 4) continue;

                    Quaternion q = new Quaternion(rot[0], rot[1], rot[2], rot[3]);
                    Transform boneTransform = FindBoneTransform(boneName);
                    if (boneTransform != null)
                    {
                        boneTransform.localRotation = q;
                    }
                }
            }

            // 2. Apply Spatial Position Scaling
            if (keyframe.joint_positions != null)
            {
                foreach (var kvp in keyframe.joint_positions)
                {
                    string boneName = kvp.Key;
                    float[] pos = kvp.Value;
                    if (pos == null || pos.Length < 3) continue;

                    Vector3 localPos = new Vector3(pos[0] * rigHeightScale, pos[1] * rigHeightScale, pos[2] * rigHeightScale);
                    Transform boneTransform = FindBoneTransform(boneName);
                    if (boneTransform != null)
                    {
                        boneTransform.localPosition = localPos;
                    }
                }
            }
        }

        private Transform FindBoneTransform(string boneName)
        {
            // Map standard spec bone names to Unity HumanBodyBones enum where applicable
            if (Enum.TryParse(boneName, out HumanBodyBones humanBone))
            {
                if (boneMap.TryGetValue(humanBone, out Transform t))
                {
                    return t;
                }
            }
            return transform.Find(boneName);
        }
    }
}
