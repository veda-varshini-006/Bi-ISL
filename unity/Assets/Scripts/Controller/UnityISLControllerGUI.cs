// Unity OnGUI Playback Control GUI Panel (Prompt 77)
// Exposes interactive GUI controls: Play, Pause, Repeat, Speed, Step-Through, Seek, and Debug Mode.

using UnityEngine;

namespace BiISL.Avatar.Controller
{
    public class UnityISLControllerGUI : MonoBehaviour
    {
        public UnityISLAnimationController controller;

        private void Awake()
        {
            if (controller == null)
            {
                controller = GetComponent<UnityISLAnimationController>();
            }
        }

        private void OnGUI()
        {
            if (controller == null) return;

            GUILayout.BeginArea(new Rect(Screen.width - 320, 20, 300, 320), "ISL Avatar Playback API", GUI.skin.window);

            // 1. Play & Pause Buttons
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("PLAY", GUILayout.Height(30)))
            {
                controller.Play();
            }
            if (GUILayout.Button("PAUSE", GUILayout.Height(30)))
            {
                controller.Pause();
            }
            GUILayout.EndHorizontal();

            // 2. Step-Through Controls
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("STEP BACK (◄)", GUILayout.Height(25)))
            {
                controller.StepBackward();
            }
            if (GUILayout.Button("STEP FWD (►)", GUILayout.Height(25)))
            {
                controller.StepForward();
            }
            GUILayout.EndHorizontal();

            // 3. Repeat Toggle
            bool repeat = GUILayout.Toggle(controller.isRepeatEnabled, " Repeat Loop");
            if (repeat != controller.isRepeatEnabled)
            {
                controller.SetRepeat(repeat);
            }

            // 4. Playback Speed Control
            GUILayout.Label($"Playback Speed: {controller.playbackSpeed:F2}x");
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("0.5x")) controller.SetSpeed(0.5f);
            if (GUILayout.Button("1.0x")) controller.SetSpeed(1.0f);
            if (GUILayout.Button("1.5x")) controller.SetSpeed(1.5f);
            if (GUILayout.Button("2.0x")) controller.SetSpeed(2.0f);
            GUILayout.EndHorizontal();

            // 5. Frame Seek Slider
            GUILayout.Space(10);
            GUILayout.Label($"Frame Seek: {controller.currentFrameIndex}");
            float newFrame = GUILayout.HorizontalSlider(controller.currentFrameIndex, 0, 100);
            if (Mathf.RoundToInt(newFrame) != controller.currentFrameIndex)
            {
                controller.SeekToFrame(Mathf.RoundToInt(newFrame));
            }

            // 6. Debug Mode Toggle
            GUILayout.Space(10);
            bool debug = GUILayout.Toggle(controller.isDebugModeEnabled, " Enable Debug Overlay (NMM HUD)");
            if (debug != controller.isDebugModeEnabled)
            {
                controller.SetDebugMode(debug);
            }

            GUILayout.EndArea();
        }
    }
}
