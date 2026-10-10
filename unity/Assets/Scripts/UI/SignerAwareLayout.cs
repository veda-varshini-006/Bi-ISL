using System;
using UnityEngine;
using BiISL.AR.Networking;

namespace BiISL.AR.UI
{
    /// <summary>
    /// Dynamically repositions the AR Subtitle HUD to prevent occluding
    /// the signer's face or active hand gestures (Requirement 7).
    /// </summary>
    [RequireComponent(typeof(RectTransform))]
    public class SignerAwareLayout : MonoBehaviour
    {
        [Header("Adaptive Positioning Settings")]
        [SerializeField] private bool isSignerAwareEnabled = true;
        [SerializeField] private float smoothSpeed = 6.0f;
        [SerializeField] private float topAnchorY = 0.82f;
        [SerializeField] private float bottomAnchorY = 0.14f;

        [Header("Thresholds")]
        [Tooltip("If hands/body extend below this normalized viewport threshold, move subtitle to top.")]
        [SerializeField] private float lowerOcclusionThreshold = 0.65f;

        private RectTransform _rectTransform;
        private Vector2 _targetAnchorMin;
        private Vector2 _targetAnchorMax;

        public bool IsSignerAwareEnabled
        {
            get => isSignerAwareEnabled;
            set
            {
                isSignerAwareEnabled = value;
                if (!value) ResetToBottomDefault();
            }
        }

        private void Awake()
        {
            _rectTransform = GetComponent<RectTransform>();
            ResetToBottomDefault();
        }

        private void OnEnable()
        {
            BiISLBackendClient.OnSubtitlePacketReceived += HandleSignerBounds;
        }

        private void OnDisable()
        {
            BiISLBackendClient.OnSubtitlePacketReceived -= HandleSignerBounds;
        }

        private void Update()
        {
            if (!isSignerAwareEnabled) return;

            // Smoothly interpolate anchor positions
            _rectTransform.anchorMin = Vector2.Lerp(_rectTransform.anchorMin, _targetAnchorMin, Time.deltaTime * smoothSpeed);
            _rectTransform.anchorMax = Vector2.Lerp(_rectTransform.anchorMax, _targetAnchorMax, Time.deltaTime * smoothSpeed);
        }

        private void HandleSignerBounds(ARSubtitlePacket packet)
        {
            if (!isSignerAwareEnabled || packet == null || packet.signer_bounds == null || !packet.signer_bounds.detected)
            {
                return;
            }

            var bounds = packet.signer_bounds;

            // Check if active gestures are occurring in lower portion of viewport
            if (bounds.max_y > lowerOcclusionThreshold && packet.is_signing)
            {
                // Subtitle card shifts to top anchor (above signer's head)
                _targetAnchorMin = new Vector2(0.05f, topAnchorY - 0.08f);
                _targetAnchorMax = new Vector2(0.95f, topAnchorY + 0.08f);
            }
            else
            {
                // Subtitle card shifts to bottom anchor
                ResetToBottomDefault();
            }
        }

        private void ResetToBottomDefault()
        {
            _targetAnchorMin = new Vector2(0.05f, bottomAnchorY - 0.06f);
            _targetAnchorMax = new Vector2(0.95f, bottomAnchorY + 0.06f);
        }
    }
}
