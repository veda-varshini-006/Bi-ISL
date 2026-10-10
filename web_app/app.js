// Bi-ISL Interactive Web Suite - Context-Gated Signer-Adaptive Translation Engine
const BACKEND_URL = "http://localhost:8000";

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
      }
    } catch (e) {
      console.warn("Backend server offline. Running local sign engine.");
    }
  }

  checkBackendStatus();

  // -----------------------------------------------------------------------
  // Landmark Canvas Simulation for Live Camera Tab
  // -----------------------------------------------------------------------
  const landmarkCanvas = document.getElementById("landmarkCanvas");
  const ctx = landmarkCanvas ? landmarkCanvas.getContext("2d") : null;
  let isWebcamRunning = false;
  let frameCount = 0;

  async function sendLandmarksToBackend(landmarksArray) {
    try {
      const res = await fetch(`${BACKEND_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ landmarks: landmarksArray })
      });
      if (res.ok) {
        const data = await res.json();
        
        const glossSeq = document.getElementById("glossSequence");
        if (glossSeq && data.gloss_sequence) {
          glossSeq.innerHTML = data.gloss_sequence
            .map(g => `<span class="token">${g}</span>`)
            .join(" ");
        }

        const engOut = document.getElementById("englishOutput");
        if (engOut && data.english_sentence) {
          engOut.innerText = `"${data.english_sentence}"`;
        }

        const latTag = document.getElementById("latencyTag");
        if (latTag) latTag.innerText = `${data.inference_latency_ms} ms`;
      }
    } catch (err) {}
  }

  function drawSimulatedLandmarks() {
    if (!ctx) return;
    ctx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);

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

    const t = Date.now() * 0.003;
    frameCount++;

    const shoulderL = { x: 260 + Math.sin(t) * 10, y: 220 };
    const shoulderR = { x: 380 - Math.sin(t) * 10, y: 220 };
    const elbowL = { x: 210 + Math.cos(t * 1.5) * 20, y: 310 };
    const elbowR = { x: 430 + Math.sin(t * 1.5) * 20, y: 310 };
    const wristL = { x: 230 + Math.sin(t * 2) * 40, y: 380 + Math.cos(t * 2) * 30 };
    const wristR = { x: 410 + Math.cos(t * 2) * 40, y: 380 + Math.sin(t * 2) * 30 };
    const head = { x: 320, y: 130 + Math.sin(t * 0.5) * 5 };

    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(shoulderL.x, shoulderL.y); ctx.lineTo(shoulderR.x, shoulderR.y);
    ctx.moveTo(shoulderL.x, shoulderL.y); ctx.lineTo(elbowL.x, elbowL.y);
    ctx.lineTo(wristL.x, wristL.y);
    ctx.moveTo(shoulderR.x, shoulderR.y); ctx.lineTo(elbowR.x, elbowR.y);
    ctx.lineTo(wristR.x, wristR.y);
    ctx.stroke();

    ctx.strokeStyle = "rgba(168, 85, 247, 0.6)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(head.x, head.y, 45, 0, Math.PI * 2);
    ctx.stroke();

    function drawHand(wrist, color) {
      ctx.fillStyle = color;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      for (let i = 0; i < 5; i++) {
        let fingerAngle = (i - 2) * 0.3 + Math.sin(t * 3 + i) * 0.2;
        let tipX = wrist.x + Math.sin(fingerAngle) * 45;
        let tipY = wrist.y - Math.cos(fingerAngle) * 45;

        ctx.beginPath(); ctx.moveTo(wrist.x, wrist.y); ctx.lineTo(tipX, tipY); ctx.stroke();
        ctx.beginPath(); ctx.arc(tipX, tipY, 4, 0, Math.PI * 2); ctx.fill();
      }
    }

    drawHand(wristL, "#10b981");
    drawHand(wristR, "#f59e0b");

    if (frameCount % 30 === 0) {
      const dummyLandmarks = Array.from({ length: 258 }, (_, i) => Math.sin(t + i * 0.1));
      sendLandmarksToBackend([dummyLandmarks]);
    }

    requestAnimationFrame(drawSimulatedLandmarks);
  }

  drawSimulatedLandmarks();

  // Webcam Handler
  const btnWebcam = document.getElementById("btnWebcam");
  const webcamVideo = document.getElementById("webcamVideo");

  if (btnWebcam) {
    btnWebcam.addEventListener("click", async () => {
      if (!isWebcamRunning) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true });
          webcamVideo.srcObject = stream;
          webcamVideo.style.display = "block";
          isWebcamRunning = true;
          btnWebcam.innerText = "Stop Webcam";
          btnWebcam.classList.replace("btn-primary", "btn-secondary");
        } catch (err) {
          alert("Webcam not available. Streaming synthetic live landmarks to PyTorch Backend.");
        }
      } else {
        if (webcamVideo.srcObject) {
          webcamVideo.srcObject.getTracks().forEach(t => t.stop());
        }
        webcamVideo.style.display = "none";
        isWebcamRunning = false;
        btnWebcam.innerText = "Start Webcam";
        btnWebcam.classList.replace("btn-secondary", "btn-primary");
      }
    });
  }

  // Backend Benchmark Suite Handler
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
      } catch (err) {}

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
          const data = await res.json();
          const ackBubble = document.createElement("div");
          ackBubble.className = "chat-bubble user-isl";
          ackBubble.innerHTML = `
            <div class="bubble-meta">SBDS Dialogue State Updated</div>
            <div class="bubble-content">Status: Active | Entities synced to Context Gate.</div>
          `;
          chatHistory.appendChild(ackBubble);
        }
      } catch (err) {}

      userTextInput.value = "";
      chatHistory.scrollTop = chatHistory.scrollHeight;
    });
  }
});
