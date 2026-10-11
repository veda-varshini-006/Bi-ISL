// Bi-ISL Interactive Web Suite - Real-Time Continuous ISL Translation Engine
const BACKEND_URL = (typeof window !== "undefined" && window.location && window.location.origin && window.location.origin.startsWith("http"))
  ? window.location.origin
  : "http://localhost:8000";

document.addEventListener("DOMContentLoaded", () => {
  // Navigation Tabs
  const navItems = document.querySelectorAll(".sidebar .nav-item");
  const tabPanes = document.querySelectorAll(".tab-pane");

  navItems.forEach(item => {
    item.addEventListener("click", () => {
      const tabId = item.getAttribute("data-tab");
      navItems.forEach(nav => nav.classList.remove("active"));
      tabPanes.forEach(pane => pane.classList.remove("active"));

      item.classList.add("active");
      const targetPane = document.getElementById(tabId);
      if (targetPane) {
        targetPane.classList.add("active");
      }
    });
  });

  // Check Backend Server Status on Load
  async function checkBackendStatus() {
    try {
      const res = await fetch(`${BACKEND_URL}/api/status`);
      if (res.ok) {
        const data = await res.json();
        const backendTag = document.getElementById("backendTag");
        if (backendTag) {
          backendTag.innerText = `Backend: PyTorch (${data.device}) - LIVE`;
        }
        const modelTag = document.getElementById("modelTag");
        if (modelTag && data.active_checkpoint) {
          modelTag.innerText = data.active_checkpoint;
        }
        const latTag = document.getElementById("latencyTag");
        if (latTag && data.model_load_time_ms) {
          latTag.innerText = `${data.model_load_time_ms} ms init`;
        }
      } else {
        const backendTag = document.getElementById("backendTag");
        if (backendTag) backendTag.innerText = "Backend: Error";
      }
    } catch (e) {
      console.warn("Backend server check:", e);
      const backendTag = document.getElementById("backendTag");
      if (backendTag) backendTag.innerText = "Backend: Offline";
    }
  }

  checkBackendStatus();

  // -----------------------------------------------------------------------
  // Live Streaming & AR Subtitle HUD Engine
  // -----------------------------------------------------------------------
  const landmarkCanvas = document.getElementById("landmarkCanvas");
  const ctx = landmarkCanvas ? landmarkCanvas.getContext("2d") : null;
  const webcamVideo = document.getElementById("webcamVideo");
  const btnWebcam = document.getElementById("btnWebcam");
  const btnSampleVideo = document.getElementById("btnSampleVideo");
  const btnUploadVideo = document.getElementById("btnUploadVideo");
  const videoFileInput = document.getElementById("videoFileInput");

  // Offscreen canvas for JPEG encoding
  const captureCanvas = document.createElement("canvas");
  captureCanvas.width = 640;
  captureCanvas.height = 480;
  const captureCtx = captureCanvas.getContext("2d");

  let isStreamingActive = false;
  let activeStreamSource = "none"; // 'webcam', 'sample', 'upload'
  let ws = null;
  let streamInterval = null;
  let latestSubtitleData = {
    active_text: "",
    status: "IDLE",
    is_finalized: false,
    confidence: 0.0,
    signer_bounds: null,
    latency_ms: 0.0
  };

  function getWebSocketUrl() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host || "localhost:8000";
    return `${protocol}//${host}/ws/ar_stream`;
  }

  function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const url = getWebSocketUrl();
    try {
      ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        console.log("[WebSocket] Connected to Bi-ISL AR streaming backend:", url);
        const backendTag = document.getElementById("backendTag");
        if (backendTag) backendTag.innerText = "Backend: WebSocket Stream - LIVE";
      };

      ws.onmessage = (event) => {
        try {
          const packet = JSON.parse(event.data);
          handleSubtitlePacket(packet);
        } catch (e) {
          console.error("[WebSocket] Packet error:", e);
        }
      };

      ws.onclose = () => {
        console.log("[WebSocket] Disconnected from streaming backend.");
      };

      ws.onerror = (err) => {
        console.warn("[WebSocket] Connection error:", err);
      };
    } catch (err) {
      console.error("[WebSocket] Failed to initialize:", err);
    }
  }

  function handleSubtitlePacket(data) {
    latestSubtitleData = data;

    const engOut = document.getElementById("englishOutput");
    if (engOut && data.active_text) {
      engOut.innerText = `"${data.active_text}"`;
    }

    const glossSeq = document.getElementById("glossSequence");
    if (glossSeq && data.active_text) {
      const tokens = data.active_text.split(" ").filter(t => t.length > 0);
      glossSeq.innerHTML = tokens
        .map(g => `<span class="token">${g}</span>`)
        .join(" ");
    }

    const latTag = document.getElementById("latencyTag");
    if (latTag && data.latency_ms) {
      latTag.innerText = `${data.latency_ms} ms`;
    }

    const gateBar = document.getElementById("gateProgressBar");
    const gateText = document.getElementById("gateValText");
    if (gateBar && gateText) {
      const pct = Math.min(100, Math.max(10, Math.round(data.confidence * 100)));
      gateBar.style.width = `${pct}%`;
      gateText.innerText = `${pct}% Visual Driven`;
    }
  }

  function startFrameCapture() {
    if (streamInterval) clearInterval(streamInterval);

    // Send binary JPEG frame every 100 ms (~10 FPS to backend)
    streamInterval = setInterval(() => {
      if (!isStreamingActive || !webcamVideo || webcamVideo.paused || webcamVideo.ended) {
        return;
      }

      if (ws && ws.readyState === WebSocket.OPEN && webcamVideo.videoWidth > 0) {
        captureCtx.drawImage(webcamVideo, 0, 0, captureCanvas.width, captureCanvas.height);
        captureCanvas.toBlob((blob) => {
          if (blob && ws && ws.readyState === WebSocket.OPEN) {
            blob.arrayBuffer().then((buffer) => {
              ws.send(buffer);
            });
          }
        }, "image/jpeg", 0.75);
      }
    }, 100);
  }

  function stopStreaming() {
    isStreamingActive = false;
    activeStreamSource = "none";
    if (streamInterval) {
      clearInterval(streamInterval);
      streamInterval = null;
    }

    if (webcamVideo.srcObject) {
      webcamVideo.srcObject.getTracks().forEach(t => t.stop());
      webcamVideo.srcObject = null;
    }
    webcamVideo.pause();
    webcamVideo.src = "";

    if (btnWebcam) {
      btnWebcam.innerText = "Start Live Camera";
      btnWebcam.classList.replace("btn-secondary", "btn-primary");
    }
    if (btnSampleVideo) {
      btnSampleVideo.innerText = "Play Sample Sign Video";
      btnSampleVideo.classList.replace("btn-primary", "btn-secondary");
    }
  }

  // Render loop on HTML5 canvas
  function renderDisplay() {
    if (!ctx) return;

    if (isStreamingActive && webcamVideo.videoWidth > 0) {
      // 1. Draw live camera / video frame
      ctx.drawImage(webcamVideo, 0, 0, landmarkCanvas.width, landmarkCanvas.height);

      // 2. Draw Signer Bounding Box if detected
      if (latestSubtitleData.signer_bounds && latestSubtitleData.signer_bounds.detected) {
        const b = latestSubtitleData.signer_bounds;
        const x = b.min_x * landmarkCanvas.width;
        const y = b.min_y * landmarkCanvas.height;
        const w = (b.max_x - b.min_x) * landmarkCanvas.width;
        const h = (b.max_y - b.min_y) * landmarkCanvas.height;

        ctx.strokeStyle = "rgba(16, 185, 129, 0.8)";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = "rgba(16, 185, 129, 0.85)";
        ctx.fillRect(x, Math.max(0, y - 20), 100, 20);
        ctx.fillStyle = "#fff";
        ctx.font = "bold 11px Inter, sans-serif";
        ctx.fillText("SIGNER DETECTED", x + 6, Math.max(14, y - 6));
      }

      // 3. Draw Floating AR Subtitle HUD
      const text = latestSubtitleData.active_text;
      const status = latestSubtitleData.status || "IDLE";

      if (text && text.trim().length > 0) {
        const isFinal = (status === "FINALIZED" || latestSubtitleData.is_finalized);
        const accentColor = isFinal ? "#39FF14" : "#00E5FF";
        const statusLabel = isFinal ? "[FINALIZED]" : "[PROVISIONAL]";

        ctx.save();
        ctx.font = "bold 18px Inter, sans-serif";
        const textMetrics = ctx.measureText(text);
        const boxWidth = Math.min(landmarkCanvas.width - 40, textMetrics.width + 60);
        const boxHeight = 52;
        const boxX = (landmarkCanvas.width - boxWidth) / 2;
        const boxY = landmarkCanvas.height - 80;

        // Dark pill background
        ctx.fillStyle = "rgba(10, 15, 29, 0.88)";
        ctx.strokeStyle = accentColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(boxX, boxY, boxWidth, boxHeight, 14);
        ctx.fill();
        ctx.stroke();

        // Status tag
        ctx.fillStyle = accentColor;
        ctx.font = "bold 11px Inter, sans-serif";
        ctx.fillText(statusLabel, boxX + 16, boxY + 20);

        // Subtitle text
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 16px Inter, sans-serif";
        ctx.fillText(text, boxX + 16, boxY + 40, boxWidth - 32);

        ctx.restore();
      }
    } else {
      // Idle / standby preview canvas with subtle grid and instructional banner
      ctx.fillStyle = "#090d16";
      ctx.fillRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);

      ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
      ctx.lineWidth = 1;
      for (let x = 0; x < landmarkCanvas.width; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, landmarkCanvas.height); ctx.stroke();
      }
      for (let y = 0; y < landmarkCanvas.height; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(landmarkCanvas.width, y); ctx.stroke();
      }

      // Standby Banner
      ctx.save();
      ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
      ctx.font = "bold 16px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Bi-ISL AR Translation Ready", landmarkCanvas.width / 2, landmarkCanvas.height / 2 - 15);

      ctx.fillStyle = "#38bdf8";
      ctx.font = "13px Inter, sans-serif";
      ctx.fillText("Click 'Start Live Camera' or 'Play Sample Sign Video' above", landmarkCanvas.width / 2, landmarkCanvas.height / 2 + 15);
      ctx.restore();
    }

    requestAnimationFrame(renderDisplay);
  }

  renderDisplay();

  // 1. Live Camera Button Handler
  if (btnWebcam) {
    btnWebcam.addEventListener("click", async () => {
      if (isStreamingActive && activeStreamSource === "webcam") {
        stopStreaming();
      } else {
        stopStreaming();
        connectWebSocket();

        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false
          });
          webcamVideo.srcObject = stream;
          await webcamVideo.play();
          isStreamingActive = true;
          activeStreamSource = "webcam";
          btnWebcam.innerText = "Stop Camera";
          btnWebcam.classList.replace("btn-primary", "btn-secondary");
          startFrameCapture();
        } catch (err) {
          alert("Could not access camera: " + err.message + "\nYou can test with 'Play Sample Sign Video' instead!");
        }
      }
    });
  }

  // 2. Play Sample Video Handler
  if (btnSampleVideo) {
    btnSampleVideo.addEventListener("click", async () => {
      if (isStreamingActive && activeStreamSource === "sample") {
        stopStreaming();
      } else {
        stopStreaming();
        connectWebSocket();

        webcamVideo.src = `${BACKEND_URL}/api/sample_video`;
        webcamVideo.loop = true;
        webcamVideo.muted = true;
        try {
          await webcamVideo.play();
          isStreamingActive = true;
          activeStreamSource = "sample";
          btnSampleVideo.innerText = "Stop Sample Video";
          btnSampleVideo.classList.replace("btn-secondary", "btn-primary");
          startFrameCapture();
        } catch (err) {
          alert("Error loading sample video: " + err.message);
        }
      }
    });
  }

  // 3. Upload Video Handler
  if (btnUploadVideo && videoFileInput) {
    btnUploadVideo.addEventListener("click", () => {
      videoFileInput.click();
    });

    videoFileInput.addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      stopStreaming();
      connectWebSocket();

      const objectUrl = URL.createObjectURL(file);
      webcamVideo.src = objectUrl;
      webcamVideo.loop = true;
      webcamVideo.muted = true;
      try {
        await webcamVideo.play();
        isStreamingActive = true;
        activeStreamSource = "upload";
        btnUploadVideo.innerText = `Playing: ${file.name.substring(0, 15)}...`;
        startFrameCapture();
      } catch (err) {
        alert("Error playing uploaded video: " + err.message);
      }
    });
  }

  // -----------------------------------------------------------------------
  // Benchmark Suite & SBDS Handlers
  // -----------------------------------------------------------------------
  const btnRunBenchmark = document.getElementById("btnRunBenchmark");
  if (btnRunBenchmark) {
    btnRunBenchmark.addEventListener("click", async () => {
      btnRunBenchmark.innerText = "⏳ Running Hardware Benchmarks...";
      btnRunBenchmark.disabled = true;

      try {
        const res = await fetch(`${BACKEND_URL}/api/benchmark`, { method: "POST" });
        if (res.ok) {
          const benchData = await res.json();
          document.getElementById("p50Val").innerText = `${benchData.p50_latency_ms} ms`;
          document.getElementById("p95Val").innerText = `${benchData.p95_latency_ms} ms (< 200 ms target)`;
          document.getElementById("memVal").innerText = `${benchData.memory_ram_mb} MB`;
          btnRunBenchmark.innerText = `✅ Benchmark Completed! (p95: ${benchData.p95_latency_ms}ms)`;
        }
      } catch (err) {
        console.error(err);
      }

      setTimeout(() => {
        btnRunBenchmark.innerText = "🚀 Run Hardware Benchmark Suite";
        btnRunBenchmark.disabled = false;
      }, 3000);
    });
  }

  // SBDS Context Input in Chat Tab
  const btnSendInput = document.getElementById("btnSendInput");
  const userTextInput = document.getElementById("userTextInput");
  const chatHistory = document.getElementById("chatHistory");

  if (btnSendInput && userTextInput && chatHistory) {
    btnSendInput.addEventListener("click", async () => {
      const text = userTextInput.value.trim();
      if (!text) return;

      const userBubble = document.createElement("div");
      userBubble.className = "chat-bubble agent-eng";
      userBubble.innerHTML = `
        <div class="bubble-meta">Dialogue Context Update (English)</div>
        <div class="bubble-content">"${text}"</div>
      `;
      chatHistory.appendChild(userBubble);

      try {
        const res = await fetch(`${BACKEND_URL}/api/sbds`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ entities: text.split(" ") })
        });

        if (res.ok) {
          const ackBubble = document.createElement("div");
          ackBubble.className = "chat-bubble user-isl";
          ackBubble.innerHTML = `
            <div class="bubble-meta">SBDS Dialogue State Updated</div>
            <div class="bubble-content">Status: Active | Entities synced to Context Gate.</div>
          `;
          chatHistory.appendChild(ackBubble);
        }
      } catch (err) {
        console.error(err);
      }

      userTextInput.value = "";
      chatHistory.scrollTop = chatHistory.scrollHeight;
    });
  }
});
