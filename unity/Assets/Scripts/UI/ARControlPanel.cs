using System;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using BiISL.AR.Camera;
using BiISL.AR.Networking;

namespace BiISL.AR.UI
{
    /// <summary>
    /// Controls for translation streaming, subtitle font sizing, caption visibility,
    /// and local Wi-Fi backend configuration (Requirement 8).
    /// </summary>
    public class ARControlPanel : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private ARCameraBackgroundController cameraController;
        [SerializeField] private BiISLBackendClient backendClient;
        [SerializeField] private ARSubtitleHUD subtitleHud;
        [SerializeField] private SignerAwareLayout signerAwareLayout;

        [Header("UI Control Buttons & Toggles")]
        [SerializeField] private Button toggleTranslationBtn;
        [SerializeField] private TextMeshProUGUI toggleTranslationBtnText;
        [SerializeField] private Button toggleCaptionsVisibilityBtn;
        [SerializeField] private TextMeshProUGUI captionsVisibilityBtnText;
        [SerializeField] private Slider fontSizeSlider;
        [SerializeField] private Toggle signerAwareToggle;

        [Header("Settings & IP Modal")]
        [SerializeField] private GameObject settingsPanel;
        [SerializeField] private TMP_InputField ipAddressInput;
        [SerializeField] private TMP_InputField portInput;
        [SerializeField] private Button applySettingsBtn;
        [SerializeField] private Button openSettingsBtn;
        [SerializeField] private Button closeSettingsBtn;

        private bool _isTranslating = true;
        private bool _isCaptionsVisible = true;

        private void Start()
        {
            InitializeControls();
        }

        private void InitializeControls()
        {
            // Start / Stop Translation
            if (toggleTranslationBtn != null)
            {
                toggleTranslationBtn.onClick.AddListener(OnToggleTranslationClicked);
            }

            // Caption Visibility Toggle
            if (toggleCaptionsVisibilityBtn != null)
            {
                toggleCaptionsVisibilityBtn.onClick.AddListener(OnToggleCaptionsVisibilityClicked);
            }

            // Font Size Slider (Range 20 to 42)
            if (fontSizeSlider != null)
            {
                fontSizeSlider.minValue = 20f;
                fontSizeSlider.maxValue = 42f;
                fontSizeSlider.value = 28f;
                fontSizeSlider.onValueChanged.AddListener(OnFontSizeChanged);
            }

            // Signer-Aware Toggle
            if (signerAwareToggle != null && signerAwareLayout != null)
            {
                signerAwareToggle.isOn = signerAwareLayout.IsSignerAwareEnabled;
                signerAwareToggle.onValueChanged.AddListener((val) => signerAwareLayout.IsSignerAwareEnabled = val);
            }

            // Settings Modal & IP Config
            if (openSettingsBtn != null)
            {
                openSettingsBtn.onClick.AddListener(() => SetSettingsVisible(true));
            }
            if (closeSettingsBtn != null)
            {
                closeSettingsBtn.onClick.AddListener(() => SetSettingsVisible(false));
            }
            if (applySettingsBtn != null)
            {
                applySettingsBtn.onClick.AddListener(OnApplySettingsClicked);
            }

            if (ipAddressInput != null && backendClient != null)
            {
                ipAddressInput.text = backendClient.CurrentHost;
            }
            if (portInput != null && backendClient != null)
            {
                portInput.text = backendClient.CurrentPort.ToString();
            }

            UpdateUIState();
        }

        private void OnToggleTranslationClicked()
        {
            _isTranslating = !_isTranslating;
            if (cameraController != null)
            {
                cameraController.IsStreamingActive = _isTranslating;
            }
            UpdateUIState();
        }

        private void OnToggleCaptionsVisibilityClicked()
        {
            _isCaptionsVisible = !_isCaptionsVisible;
            if (subtitleHud != null)
            {
                subtitleHud.IsCaptionsVisible = _isCaptionsVisible;
            }
            UpdateUIState();
        }

        private void OnFontSizeChanged(float size)
        {
            if (subtitleHud != null)
            {
                subtitleHud.SetFontSize(size);
            }
        }

        private void OnApplySettingsClicked()
        {
            if (backendClient != null && ipAddressInput != null)
            {
                string host = ipAddressInput.text.Trim();
                int port = 8000;
                if (portInput != null && int.TryParse(portInput.text.Trim(), out int parsedPort))
                {
                    port = parsedPort;
                }
                backendClient.SetServerConfig(host, port);
            }
            SetSettingsVisible(false);
        }

        private void SetSettingsVisible(bool visible)
        {
            if (settingsPanel != null)
            {
                settingsPanel.SetActive(visible);
            }
        }

        private void UpdateUIState()
        {
            if (toggleTranslationBtnText != null)
            {
                toggleTranslationBtnText.text = _isTranslating ? "Pause Recognition" : "Resume Recognition";
            }
            if (captionsVisibilityBtnText != null)
            {
                captionsVisibilityBtnText.text = _isCaptionsVisible ? "Hide Captions" : "Show Captions";
            }
        }
    }
}
