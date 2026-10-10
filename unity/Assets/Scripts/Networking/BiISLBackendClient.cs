using System;
using System.Collections;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;
using BiISL.AR.Camera;

namespace BiISL.AR.Networking
{
    public enum ConnectionState
    {
        Disconnected,
        Connecting,
        Connected,
        Reconnecting,
        Error
    }

    [Serializable]
    public class SignerBounds
    {
        public float min_x;
        public float min_y;
        public float max_x;
        public float max_y;
        public bool detected;
    }

    [Serializable]
    public class ARSubtitlePacket
    {
        public string active_text;
        public string status; // "PROVISIONAL", "FINALIZED", "IDLE"
        public float confidence;
        public bool is_signing;
        public SignerBounds signer_bounds;
        public float latency_ms;
        public int buffered_frames;
    }

    /// <summary>
    /// Handles bidirectional WebSocket & HTTP streaming communication with the Python FastAPI backend.
    /// </summary>
    public class BiISLBackendClient : MonoBehaviour
    {
        [Header("Backend Wi-Fi Configuration")]
        [SerializeField] private string serverHost = "192.168.1.5";
        [SerializeField] private int serverPort = 8000;
        [SerializeField] private bool useWebSocket = true;

        [Header("Resilience Settings")]
        [SerializeField] private float reconnectIntervalSec = 3.0f;
        [SerializeField] private float heartbeatIntervalSec = 5.0f;

        public static event Action<ARSubtitlePacket> OnSubtitlePacketReceived;
        public static event Action<ConnectionState, string> OnConnectionStatusChanged;

        private ClientWebSocket _webSocket;
        private CancellationTokenSource _cts;
        private ConnectionState _currentState = ConnectionState.Disconnected;
        private bool _isDisposed;
        private float _lastHeartbeatTime;

        public ConnectionState CurrentState => _currentState;
        public string CurrentHost => serverHost;
        public int CurrentPort => serverPort;

        public string HttpUrl => $"http://{serverHost}:{serverPort}";
        public string WebSocketUrl => $"ws://{serverHost}:{serverPort}/ws/ar_stream";

        private void OnEnable()
        {
            ARCameraBackgroundController.OnFrameEncoded += HandleFrameEncoded;
        }

        private void OnDisable()
        {
            ARCameraBackgroundController.OnFrameEncoded -= HandleFrameEncoded;
            Disconnect();
        }

        private void Start()
        {
            Connect();
        }

        public void SetServerConfig(string host, int port)
        {
            serverHost = host.Trim();
            serverPort = port;
            Disconnect();
            Connect();
        }

        public async void Connect()
        {
            if (_currentState == ConnectionState.Connected || _currentState == ConnectionState.Connecting)
            {
                return;
            }

            SetState(ConnectionState.Connecting, $"Connecting to {serverHost}:{serverPort}...");
            _cts?.Cancel();
            _cts = new CancellationTokenSource();

            if (useWebSocket)
            {
                await ConnectWebSocketAsync();
            }
            else
            {
                StartCoroutine(CheckHttpHealthCoroutine());
            }
        }

        public void Disconnect()
        {
            _cts?.Cancel();
            if (_webSocket != null)
            {
                if (_webSocket.State == WebSocketState.Open)
                {
                    _webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Client closing", CancellationToken.None);
                }
                _webSocket.Dispose();
                _webSocket = null;
            }
            SetState(ConnectionState.Disconnected, "Disconnected");
        }

        private async Task ConnectWebSocketAsync()
        {
            try
            {
                _webSocket = new ClientWebSocket();
                var uri = new Uri(WebSocketUrl);
                await _webSocket.ConnectAsync(uri, _cts.Token);

                SetState(ConnectionState.Connected, "Connected to Bi-ISL Engine");
                _ = ReceiveLoopAsync();
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[Bi-ISL AR] WebSocket connection failed: {ex.Message}");
                SetState(ConnectionState.Error, $"Connection failed: {ex.Message}");
                ScheduleReconnect();
            }
        }

        private async Task ReceiveLoopAsync()
        {
            var buffer = new byte[8192];
            var messageBuilder = new StringBuilder();

            while (_webSocket != null && _webSocket.State == WebSocketState.Open && !_cts.IsCancellationRequested)
            {
                try
                {
                    var result = await _webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), _cts.Token);

                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        await _webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closed by server", CancellationToken.None);
                        SetState(ConnectionState.Disconnected, "Server closed connection");
                        ScheduleReconnect();
                        break;
                    }

                    messageBuilder.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));

                    if (result.EndOfMessage)
                    {
                        string json = messageBuilder.ToString();
                        messageBuilder.Clear();
                        ParseAndDispatchPacket(json);
                    }
                }
                catch (Exception ex)
                {
                    if (!_cts.IsCancellationRequested)
                    {
                        Debug.LogWarning($"[Bi-ISL AR] WebSocket receive error: {ex.Message}");
                        SetState(ConnectionState.Reconnecting, "Connection interrupted. Reconnecting...");
                        ScheduleReconnect();
                    }
                    break;
                }
            }
        }

        private void ParseAndDispatchPacket(string json)
        {
            try
            {
                var packet = JsonUtility.FromJson<ARSubtitlePacket>(json);
                if (packet != null)
                {
                    // Dispatch to main Unity thread
                    OnSubtitlePacketReceived?.Invoke(packet);
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[Bi-ISL AR] Failed parsing subtitle packet: {ex.Message}");
            }
        }

        private void HandleFrameEncoded(byte[] jpegBytes, int width, int height)
        {
            if (_currentState != ConnectionState.Connected || _webSocket == null || _webSocket.State != WebSocketState.Open)
            {
                return;
            }

            try
            {
                _webSocket.SendAsync(
                    new ArraySegment<byte>(jpegBytes),
                    WebSocketMessageType.Binary,
                    true,
                    _cts.Token
                );
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[Bi-ISL AR] Error sending camera frame: {ex.Message}");
            }
        }

        private void ScheduleReconnect()
        {
            if (!_isDisposed && isActiveAndEnabled)
            {
                StopAllCoroutines();
                StartCoroutine(ReconnectCoroutine());
            }
        }

        private IEnumerator ReconnectCoroutine()
        {
            SetState(ConnectionState.Reconnecting, $"Reconnecting in {reconnectIntervalSec:F0}s...");
            yield return new WaitForSeconds(reconnectIntervalSec);
            Connect();
        }

        private IEnumerator CheckHttpHealthCoroutine()
        {
            using var req = UnityWebRequest.Get($"{HttpUrl}/health");
            req.timeout = 3;
            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                SetState(ConnectionState.Connected, "Connected (HTTP Mode)");
            }
            else
            {
                SetState(ConnectionState.Error, $"HTTP Health check failed: {req.error}");
                ScheduleReconnect();
            }
        }

        private void SetState(ConnectionState state, string message)
        {
            _currentState = state;
            OnConnectionStatusChanged?.Invoke(state, message);
        }
    }
}
