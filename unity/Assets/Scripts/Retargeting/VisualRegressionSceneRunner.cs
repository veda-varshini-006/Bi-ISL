// Unity Visual Regression Scene Runner (Prompt 74)
// Drives visual regression benchmark scenes, camera perspective changes, and snapshot exports.

using System.IO;
using UnityEngine;

namespace BiISL.Avatar.Retargeting
{
    public class VisualRegressionSceneRunner : MonoBehaviour
    {
        public Camera renderCamera;
        public HumanoidRetargeter retargeter;
        public string sceneConfigPath;

        public void RunSceneValidation(string jsonClipData, string outputImagePath)
        {
            if (retargeter == null || renderCamera == null)
            {
                Debug.LogError("[VisualRegressionSceneRunner] Missing required references.");
                return;
            }

            AnimationClipData clipData = JsonUtility.FromJson<AnimationClipData>(jsonClipData);
            if (clipData != null && clipData.keyframes != null && clipData.keyframes.Count > 0)
            {
                // Apply midpoint pose for baseline snapshot frame
                int midIndex = clipData.keyframes.Count / 2;
                retargeter.ApplyKeyframe(clipData.keyframes[midIndex]);

                RenderTexture rt = new RenderTexture(1024, 1024, 24);
                renderCamera.targetTexture = rt;
                Texture2D screenShot = new Texture2D(1024, 1024, TextureFormat.RGB24, false);
                renderCamera.Render();

                RenderTexture.active = rt;
                screenShot.ReadPixels(new Rect(0, 0, 1024, 1024), 0, 0);
                screenShot.Apply();

                renderCamera.targetTexture = null;
                RenderTexture.active = null;
                DestroyImmediate(rt);

                byte[] bytes = screenShot.EncodeToPNG();
                File.WriteAllBytes(outputImagePath, bytes);
                Debug.Log($"[VisualRegressionSceneRunner] Captured frame snapshot to: {outputImagePath}");
            }
        }
    }
}
