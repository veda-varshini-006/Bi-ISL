using System;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using BiISL.AR.Networking;

namespace BiISL.AR.UI
{
    /// <summary>
    /// Displays stabilized AR subtitles, confidence meters, and connection status using TextMeshPro.
    /// </summary>
    public class ARSubtitleHUD : MonoBehaviour
    {
        [Header("TextMeshPro Display Elements")]
        [SerializeField] private TextMeshProUGUI subtitleText;
        [SerializeField] private TextMeshProUGUI statusBadgeText;
        [SerializeField] private TextMeshProUGUI confidenceText;
        [SerializeField] private TextMeshProUGUI latencyText;
        [SerializeField] private TextMeshProUGUI connectionStatusText;

        [Header("Visual Indicators & Cards")]
        [SerializeField] private CanvasGroup subtitleCardCanvasGroup;
        [SerializeField] private Image statusBorderImage;
        [SerializeField] private Image confidenceFillImage;
        [SerializeField] private GameObject loadingSpinner;
        [SerializeField] private GameObject activeSigningDot;

        [Header("Aesthetic Colors")]
        [SerializeField] private Color finalizedColor = new Color(0.06f, 0.72f, 0.51f, 1f); // Emerald
        [SerializeField] private Color provisionalColor = new Color(0.96f, 0.62f, 0.04f, 1f); // Amber
        [SerializeField] private Color listeningColor = new Color(0.22f, 0.74f, 0.97f, 1f); // Sky Blue

        [Header("Display Settings")]
        [SerializeField] private float holdDurationSec = 4.5f;
        [SerializeField] private float fadeDurationSec = 0.35f;

        private float _lastUpdateTime;
        private bool _isCaptionsVisible = true;
        private Coroutine _fadeCoroutine;

        public bool IsCaptionsVisible
        {
            get => _isCaptionsVisible;
            set
            {
                _isCaptionsVisible = value;
                if (subtitleCardCanvasGroup != null)
                {
                    subtitleCardCanvasGroup.alpha = value ? 1.0f : 0.0f;
                }
            }
        }

        private void OnEnable()
        {
            BiISLBackendClient.OnSubtitlePacketReceived += HandleSubtitlePacket;
            BiISLBackendClient.OnConnectionStatusChanged += HandleConnectionStatus;
        }

        private void OnDisable()
        {
            BiISLBackendClient.OnSubtitlePacketReceived -= HandleSubtitlePacket;
            BiISLBackendClient.OnConnectionStatusChanged -= HandleConnectionStatus;
        }

        private void Start()
        {
            SetDefaultState();
        }

        private void Update()
        {
            // Auto-fade expired subtitles if no new updates arrived
            if (subtitleText != null && !string.IsNullOrEmpty(subtitleText.text))
            {
                if (Time.time - _lastUpdateTime > holdDurationSec && subtitleCardCanvasGroup != null && subtitleCardCanvasGroup.alpha > 0.05f)
                {
                    if (_fadeCoroutine == null)
                    {
                        _fadeCoroutine = StartCoroutine(FadeSubtitleCard(0.0f));
                    }
                }
            }
        }

        public void SetFontSize(float size)
        {
            if (subtitleText != null)
            {
                subtitleText.fontSize = size;
            }
        }

        private void HandleSubtitlePacket(ARSubtitlePacket packet)
        {
            _lastUpdateTime = Time.time;

            if (!_isCaptionsVisible) return;

            // Ensure card is visible
            if (subtitleCardCanvasGroup != null)
            {
                if (_fadeCoroutine != null) StopCoroutine(_fadeCoroutine);
                subtitleCardCanvasGroup.alpha = 1.0f;
            }

            // 1. Text Update
            string textToDisplay = !string.IsNullOrEmpty(packet.active_text) ? packet.active_text : "Position in frame & start signing...";
            if (subtitleText != null)
            {
                subtitleText.text = textToDisplay;
            }

            // 2. Status Badge & Colors
            Color themeColor;
            string badgeString;
            if (packet.status == "FINALIZED")
            {
                themeColor = finalizedColor;
                badgeString = "[FINALIZED]";
                if (subtitleText != null) subtitleText.color = Color.white;
            }
            elif (packet.status == "PROVISIONAL")
            {
                themeColor = provisionalColor;
                badgeString = "[PROVISIONAL]";
                if (subtitleText != null) subtitleText.color = new Color(0.9f, 0.95f, 1f, 0.9f);
            }
            else
            {
                themeColor = listeningColor;
                badgeString = "[LISTENING]";
                if (subtitleText != null) subtitleText.color = new Color(0.8f, 0.85f, 0.9f, 0.8f);
            }

            if (statusBadgeText != null)
            {
                statusBadgeText.text = badgeString;
                statusBadgeText.color = themeColor;
            }
            if (statusBorderImage != null)
            {
                statusBorderImage.color = themeColor;
            }

            // 3. Confidence Bar & Text
            if (confidenceText != null)
            {
                confidenceText.text = $"Confidence: {Mathf.RoundToInt(packet.confidence * 100)}%";
            }
            if (confidenceFillImage != null)
            {
                confidenceFillImage.fillAmount = Mathf.Clamp01(packet.confidence);
                confidenceFillImage.color = themeColor;
            }

            // 4. Latency
            if (latencyText != null)
            {
                latencyText.text = $"{packet.latency_ms:F1} ms";
            }

            // 5. Active Signing Indicator
            if (activeSigningDot != null)
            {
                activeSigningDot.SetActive(packet.is_signing);
            }
        }

        private void HandleConnectionStatus(ConnectionState state, string message)
        {
            if (connectionStatusText != null)
            {
                connectionStatusText.text = message;
            }

            if (loadingSpinner != null)
            {
                loadingSpinner.SetActive(state == ConnectionState.Connecting || state == ConnectionState.Reconnecting);
            }
        }

        private void SetDefaultState()
        {
            if (subtitleText != null)
            {
                subtitleText.text = "Waiting for live camera signing...";
            }
            if (statusBadgeText != null)
            {
                statusBadgeText.text = "[INITIALIZING]";
                statusBadgeText.color = listeningColor;
            }
            if (confidenceText != null)
            {
                confidenceText.text = "Confidence: --";
            }
            if (activeSigningDot != null)
            {
                activeSigningDot.SetActive(false);
            }
        }

        private IEnumerator FadeSubtitleCard(float targetAlpha)
        {
            float startAlpha = subtitleCardCanvasGroup.alpha;
            float elapsed = 0f;

            while (elapsed < fadeDurationSec)
            {
                elapsed += Time.deltaTime;
                subtitleCardCanvasGroup.alpha = Mathf.Lerp(startAlpha, targetAlpha, elapsed / fadeDurationSec);
                yield return null;
            }

            subtitleCardCanvasGroup.alpha = targetAlpha;
            _fadeCoroutine = null;
        }
    }
}
