using System;
using System.Collections;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace BiISL.AR.Camera
{
    /// <summary>
    /// Manages AR Camera background acquisition and CPU frame extraction for ISL translation.
    /// Supports both AR Foundation XRCpuImage and Unity Editor WebCamTexture fallback.
    /// </summary>
    public class ARCameraBackgroundController : MonoBehaviour
    {
        [Header("AR Foundation Components")]
        [SerializeField] private ARCameraManager arCameraManager;
        [SerializeField] private ARCameraBackground arCameraBackground;

        [Header("Frame Streaming Settings")]
        [Range(5, 30)]
        [SerializeField] private int targetStreamingFps = 15;
        [Range(30, 90)]
        [SerializeField] private int jpegQuality = 65;
        [SerializeField] private int downsampleWidth = 480;
        [SerializeField] private int downsampleHeight = 640;

        public static event Action<byte[], int, int> OnFrameEncoded;

        private float _timeSinceLastFrame;
        private Texture2D _conversionTexture;
        private WebCamTexture _editorWebcamTexture;
        private bool _isStreamingActive = true;

        public bool IsStreamingActive
        {
            get => _isStreamingActive;
            set => _isStreamingActive = value;
        }

        private void Awake()
        {
            if (arCameraManager == null)
            {
                arCameraManager = FindFirstObjectByType<ARCameraManager>();
            }
            if (arCameraBackground == null)
            {
                arCameraBackground = FindFirstObjectByType<ARCameraBackground>();
            }
        }

        private void OnEnable()
        {
            if (arCameraManager != null)
            {
                arCameraManager.frameReceived += OnCameraFrameReceived;
            }
            else
            {
                StartEditorWebcamFallback();
            }
        }

        private void OnDisable()
        {
            if (arCameraManager != null)
            {
                arCameraManager.frameReceived -= OnCameraFrameReceived;
            }
            StopEditorWebcamFallback();
        }

        private void OnDestroy()
        {
            if (_conversionTexture != null)
            {
                Destroy(_conversionTexture);
            }
            StopEditorWebcamFallback();
        }

        private void Update()
        {
            _timeSinceLastFrame += Time.deltaTime;

            // Editor WebCamTexture streaming loop
            if (_editorWebcamTexture != null && _editorWebcamTexture.isPlaying && _isStreamingActive)
            {
                float frameInterval = 1.0f / targetStreamingFps;
                if (_timeSinceLastFrame >= frameInterval)
                {
                    _timeSinceLastFrame = 0f;
                    ProcessEditorWebcamFrame();
                }
            }
        }

        private void OnCameraFrameReceived(ARCameraFrameEventArgs eventArgs)
        {
            if (!_isStreamingActive) return;

            float frameInterval = 1.0f / targetStreamingFps;
            if (_timeSinceLastFrame < frameInterval) return;
            _timeSinceLastFrame = 0f;

            if (arCameraManager.TryAcquireLatestCpuImage(out XRCpuImage image))
            {
                using (image)
                {
                    ConvertAndDispatchXRCpuImage(image);
                }
            }
        }

        private void ConvertAndDispatchXRCpuImage(XRCpuImage image)
        {
            // Set up conversion params to target size and RGB24
            var conversionParams = new XRCpuImage.ConversionParams
            {
                inputRect = new RectInt(0, 0, image.width, image.height),
                outputDimensions = new Vector2Int(downsampleWidth, downsampleHeight),
                outputFormat = TextureFormat.RGB24,
                transformation = XRCpuImage.Transformation.MirrorY
            };

            int size = image.GetConvertedDataSize(conversionParams);
            using var rawBuffer = new NativeArray<byte>(size, Allocator.Temp);
            image.Convert(conversionParams, rawBuffer);

            if (_conversionTexture == null || _conversionTexture.width != downsampleWidth || _conversionTexture.height != downsampleHeight)
            {
                _conversionTexture = new Texture2D(downsampleWidth, downsampleHeight, TextureFormat.RGB24, false);
            }

            _conversionTexture.LoadRawTextureData(rawBuffer);
            _conversionTexture.Apply();

            byte[] jpegBytes = _conversionTexture.EncodeToJPG(jpegQuality);
            OnFrameEncoded?.Invoke(jpegBytes, downsampleWidth, downsampleHeight);
        }

        #region Editor / Non-AR Fallback

        private void StartEditorWebcamFallback()
        {
            if (Application.isEditor || !Application.isMobilePlatform)
            {
                Debug.Log("[Bi-ISL AR] ARCameraManager not active. Starting WebCamTexture fallback.");
                if (WebCamTexture.devices.Length > 0)
                {
                    _editorWebcamTexture = new WebCamTexture(640, 480, 30);
                    _editorWebcamTexture.Play();
                }
                else
                {
                    Debug.LogWarning("[Bi-ISL AR] No webcam device available.");
                }
            }
        }

        private void StopEditorWebcamFallback()
        {
            if (_editorWebcamTexture != null)
            {
                if (_editorWebcamTexture.isPlaying)
                {
                    _editorWebcamTexture.Stop();
                }
                Destroy(_editorWebcamTexture);
                _editorWebcamTexture = null;
            }
        }

        private void ProcessEditorWebcamFrame()
        {
            if (_editorWebcamTexture.width < 16) return;

            if (_conversionTexture == null || _conversionTexture.width != _editorWebcamTexture.width || _conversionTexture.height != _editorWebcamTexture.height)
            {
                _conversionTexture = new Texture2D(_editorWebcamTexture.width, _editorWebcamTexture.height, TextureFormat.RGB24, false);
            }

            _conversionTexture.SetPixels(_editorWebcamTexture.GetPixels());
            _conversionTexture.Apply();

            byte[] jpegBytes = _conversionTexture.EncodeToJPG(jpegQuality);
            OnFrameEncoded?.Invoke(jpegBytes, _conversionTexture.width, _conversionTexture.height);
        }

        #endregion
    }
}
