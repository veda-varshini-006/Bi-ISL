// Bi-ISL Interactive Web Suite - Live Backend & Side-by-Side Studio Integration
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
      document.getElementById(tabId).classList.add("active");
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
      console.warn("Backend server offline. Running simulated mode.");
    }
  }

  checkBackendStatus();

  // Side-by-Side Text-to-ISL Studio Handler
  const btnRenderStudio = document.getElementById("btnRenderStudio");
  const studioTextInput = document.getElementById("studioTextInput");
  const studioGlossDisplay = document.getElementById("studioGlossDisplay");
  const studioNmmDisplay = document.getElementById("studioNmmDisplay");
  const nowSigningTag = document.getElementById("nowSigningTag");

  let activeGlosses = ["DOCTOR", "APPOINTMENT", "TODAY", "TIME", "WHAT"];
  let currentGlossIndex = 0;

  async function handleStudioTranslation(text) {
    if (!text) return;

    try {
      const res = await fetch(`${BACKEND_URL}/api/text_to_isl`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });

      if (res.ok) {
        const data = await res.json();
        activeGlosses = data.isl_gloss_sequence;

        // Render Gloss Tokens
        if (studioGlossDisplay) {
          studioGlossDisplay.innerHTML = activeGlosses
            .map(g => `<span class="token">${g}</span>`)
            .join(" ");
        }

        // Render NMM Specs
        if (studioNmmDisplay) {
          const nmm = data.nmm_specifications;
          studioNmmDisplay.innerText = `Eyebrows: ${nmm.eyebrows} | Head Pose: ${nmm.head_tilt} | Mouthings: ${nmm.mouthings}`;
        }
      }
    } catch (err) {
      // Fallback
      activeGlosses = text.toUpperCase().split(" ").filter(w => w.length > 2);
      if (studioGlossDisplay) {
        studioGlossDisplay.innerHTML = activeGlosses
          .map(g => `<span class="token">${g}</span>`)
          .join(" ");
      }
    }

    currentGlossIndex = 0;
    if (nowSigningTag) {
      nowSigningTag.innerText = `Now Signing: [${activeGlosses[0] || 'READY'}]`;
    }
  }

  if (btnRenderStudio && studioTextInput) {
    btnRenderStudio.addEventListener("click", () => {
      handleStudioTranslation(studioTextInput.value.trim());
    });
  }

  // Preset Sentence Buttons
  const presetBtns = document.querySelectorAll(".preset-btn");
  presetBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const text = btn.getAttribute("data-text");
      if (studioTextInput) studioTextInput.value = text;
      handleStudioTranslation(text);
    });
  });

  // Landmark Canvas Simulation
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

  // 3D Avatar Rendering Canvas (Driven by Active Glosses)
  const avatarCanvas = document.getElementById("avatarCanvas");
  const actx = avatarCanvas ? avatarCanvas.getContext("2d") : null;
  let isAvatarPlaying = true;
  let signCycleTimer = 0;

  function render3DAvatar() {
    if (!actx) return;
    actx.clearRect(0, 0, avatarCanvas.width, avatarCanvas.height);

    const time = Date.now() * 0.003;
    const cx = avatarCanvas.width / 2;
    const cy = avatarCanvas.height / 2;

    // Cycle through active glosses every 1.5 seconds
    signCycleTimer++;
    if (signCycleTimer % 90 === 0 && activeGlosses.length > 0) {
      currentGlossIndex = (currentGlossIndex + 1) % activeGlosses.length;
      if (nowSigningTag) {
        nowSigningTag.innerText = `Now Signing: [${activeGlosses[currentGlossIndex]}]`;
      }
    }

    actx.fillStyle = "#1e293b";
    actx.strokeStyle = "#818cf8";
    actx.lineWidth = 3;

    actx.beginPath();
    actx.arc(cx, cy - 80 + Math.sin(time) * 4, 50, 0, Math.PI * 2);
    actx.fill();
    actx.stroke();

    actx.fillStyle = "#60a5fa";
    actx.beginPath(); actx.arc(cx - 18, cy - 90, 6, 0, Math.PI * 2); actx.fill();
    actx.beginPath(); actx.arc(cx + 18, cy - 90, 6, 0, Math.PI * 2); actx.fill();

    // Sign Gesture Motion mapped to active gloss index
    const signFrequency = 2 + (currentGlossIndex % 3) * 0.5;
    const armLX = cx - 50 + Math.cos(time * signFrequency) * 65;
    const armLY = cy + 20 + Math.sin(time * signFrequency) * 45;
    const armRX = cx + 50 + Math.sin(time * (signFrequency + 0.5)) * 65;
    const armRY = cy + 20 + Math.cos(time * (signFrequency + 0.5)) * 45;

    actx.strokeStyle = "#a7f3d0";
    actx.lineWidth = 6;
    actx.beginPath(); ctx.moveTo(cx - 45, cy - 10); ctx.lineTo(armLX, armLY); ctx.stroke();
    actx.beginPath(); ctx.moveTo(cx + 45, cy - 10); ctx.lineTo(armRX, armRY); ctx.stroke();

    actx.fillStyle = "#34d399";
    actx.beginPath(); actx.arc(armLX, armLY, 12, 0, Math.PI * 2); actx.fill();
    actx.beginPath(); actx.arc(armRX, armRY, 12, 0, Math.PI * 2); actx.fill();

    if (isAvatarPlaying) {
      requestAnimationFrame(render3DAvatar);
    }
  }

  render3DAvatar();

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

  // Text-to-ISL English Input in Chat Tab
  const btnSendInput = document.getElementById("btnSendInput");
  const userTextInput = document.getElementById("userTextInput");
  const chatHistory = document.getElementById("chatHistory");

  if (btnSendInput && userTextInput && chatHistory) {
    btnSendInput.addEventListener("click", async () => {
      const text = userTextInput.value.trim();
      if (!text) return;

      const userBubble = document.createElement("div");
      userBubble.className = "chat-bubble user-isl";
      userBubble.innerHTML = `
        <div class="bubble-meta">User Input (English)</div>
        <div class="bubble-content">"${text}"</div>
      `;
      chatHistory.appendChild(userBubble);

      try {
        const res = await fetch(`${BACKEND_URL}/api/text_to_isl`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text })
        });

        if (res.ok) {
          const data = await res.json();
          const agentBubble = document.createElement("div");
          agentBubble.className = "chat-bubble agent-eng";
          agentBubble.innerHTML = `
            <div class="bubble-meta">Backend Synthesized ISL Representation</div>
            <div class="bubble-content">Glosses: [${data.isl_gloss_sequence.join(" ")}]</div>
            <div class="bubble-trans">3D Avatar rendering sign sequence live beside text.</div>
          `;
          chatHistory.appendChild(agentBubble);

          // Update studio avatar as well
          activeGlosses = data.isl_gloss_sequence;
          currentGlossIndex = 0;
        }
      } catch (err) {}

      userTextInput.value = "";
      chatHistory.scrollTop = chatHistory.scrollHeight;
    });
  }
});
