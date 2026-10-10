using System;
using System.Collections;
using UnityEngine;
#if UNITY_ANDROID
using UnityEngine.Android;
#endif

namespace BiISL.AR.Permissions
{
    /// <summary>
    /// Handles Android runtime camera and internet permissions gracefully.
    /// </summary>
    public class AndroidPermissionHelper : MonoBehaviour
    {
        public static event Action OnCameraPermissionGranted;
        public static event Action<string> OnPermissionDenied;

        private void Start()
        {
            RequestCameraPermission();
        }

        public void RequestCameraPermission()
        {
#if UNITY_ANDROID
            if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
            {
                var callbacks = new PermissionCallbacks();
                callbacks.PermissionGranted += (permissionName) =>
                {
                    Debug.Log($"[Bi-ISL AR] Permission granted: {permissionName}");
                    OnCameraPermissionGranted?.Invoke();
                };
                callbacks.PermissionDenied += (permissionName) =>
                {
                    Debug.LogWarning($"[Bi-ISL AR] Permission denied: {permissionName}");
                    OnPermissionDenied?.Invoke("Camera permission is required for live ISL recognition.");
                };
                callbacks.PermissionDeniedAndDontAskAgain += (permissionName) =>
                {
                    Debug.LogError($"[Bi-ISL AR] Camera permission permanently denied: {permissionName}");
                    OnPermissionDenied?.Invoke("Camera permission was permanently denied. Please enable it in Android Settings.");
                };

                Permission.RequestUserPermission(Permission.Camera, callbacks);
            }
            else
            {
                OnCameraPermissionGranted?.Invoke();
            }
#else
            // In Unity Editor or standalone
            OnCameraPermissionGranted?.Invoke();
#endif
        }

        public bool HasCameraPermission()
        {
#if UNITY_ANDROID
            return Permission.HasUserAuthorizedPermission(Permission.Camera);
#else
            return true;
#endif
        }
    }
}
