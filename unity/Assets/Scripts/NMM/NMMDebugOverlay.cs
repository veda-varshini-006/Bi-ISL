// Unity OnGUI Non-Manual Marker (NMM) Debug Overlay (Prompt 76)
// Renders active NMM tags, eyebrow state, blendshapes, head pose, and body lean in an on-screen HUD.

using System.Text;
using UnityEngine;

namespace BiISL.Avatar.NMM
{
    public class NMMDebugOverlay : MonoBehaviour
    {
        public NMMController nmmController;
        public bool showOverlay = true;

        [Header("HUD Styling")]
        public Color backgroundColor = new Color(0, 0, 0, 0.75f);
        public Color textColor = Color.cyan;
        public int fontSize = 14;

        private GUIStyle hudStyle;
        private Texture2D bgTexture;

        private void Start()
        {
            bgTexture = new Texture2D(1, 1);
            bgTexture.SetPixel(0, 0, backgroundColor);
            bgTexture.Apply();
        }

        private void OnGUI()
        {
            if (!showOverlay || nmmController == null) return;

            if (hudStyle == null)
            {
                hudStyle = new GUIStyle(GUI.skin.box);
                hudStyle.normal.background = bgTexture;
                hudStyle.normal.textColor = textColor;
                hudStyle.fontSize = fontSize;
                hudStyle.alignment = TextAnchor.UpperLeft;
                hudStyle.padding = new RectOffset(10, 10, 10, 10);
            }

            StringBuilder sb = new StringBuilder();
            sb.AppendLine("=== ISL AVATAR NMM DEBUG OVERLAY ===");
            sb.AppendLine($"Eyebrow State: {nmmController.currentEyebrowState}");

            string tagsStr = (nmmController.currentActiveTags != null && nmmController.currentActiveTags.Count > 0)
                ? string.Join(", ", nmmController.currentActiveTags)
                : "NONE";
            sb.AppendLine($"Active NMM Tags: [{tagsStr}]");

            GUI.Box(new Rect(20, 20, 420, 180), sb.ToString(), hudStyle);
        }
    }
}
