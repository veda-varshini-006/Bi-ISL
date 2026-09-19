// Bi-ISL Interactive Web Suite - Subway Surfers Style 3D Boy Avatar with Ultra-Clear Animated Fingers
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
      console.warn("Backend server offline. Running local sign engine.");
    }
  }

  checkBackendStatus();

  // Active Glosses for Avatar Animation
  let activeGlosses = ["DOCTOR", "APPOINTMENT", "TODAY", "TIME", "WHAT"];
  let currentGlossIndex = 0;
  let activeNMM = { eyebrows: "RAISED", head_tilt: "LEFT", mouthings: "/dok-tor/" };

  // Side-by-Side Text-to-ISL Studio Handler
  const btnRenderStudio = document.getElementById("btnRenderStudio");
  const studioTextInput = document.getElementById("studioTextInput");
  const studioGlossDisplay = document.getElementById("studioGlossDisplay");
  const studioNmmDisplay = document.getElementById("studioNmmDisplay");
  const nowSigningTag = document.getElementById("nowSigningTag");

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
        activeNMM = data.nmm_specifications;

        if (studioGlossDisplay) {
          studioGlossDisplay.innerHTML = activeGlosses
            .map(g => `<span class="token">${g}</span>`)
            .join(" ");
        }

        if (studioNmmDisplay) {
          studioNmmDisplay.innerText = `Eyebrows: ${activeNMM.eyebrows} | Head Pose: ${activeNMM.head_tilt} | Mouthings: ${activeNMM.mouthings}`;
        }
      }
    } catch (err) {
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

  // -----------------------------------------------------------------------
  // SUBWAY SURFERS STYLE 3D BOY AVATAR WITH ULTRA-CLEAR ANIMATED FINGERS
  // -----------------------------------------------------------------------
  const avatarCanvas = document.getElementById("avatarCanvas");
  const actx = avatarCanvas ? avatarCanvas.getContext("2d") : null;
  let isAvatarPlaying = true;
  let signCycleTimer = 0;

  /**
   * Draws enlarged, ultra-clear 3D articulated fingers for Subway Surfers Boy
   */
  function drawClearSubwaySurfersHand(ctx, wristX, wristY, angle, handShape = "OPEN") {
    ctx.save();
    ctx.translate(wristX, wristY);
    ctx.rotate(angle);

    // Glowing Palm Outline for maximum visibility
    ctx.shadowColor = "#38bdf8";
    ctx.shadowBlur = 12;

    // Palm Base (Skin Gradient)
    const palmGrad = ctx.createRadialGradient(0, 0, 4, 0, 0, 22);
    palmGrad.addColorStop(0, "#ffe4e6");
    palmGrad.addColorStop(0.7, "#fb7185");
    palmGrad.addColorStop(1, "#e11d48");

    ctx.fillStyle = palmGrad;
    ctx.beginPath();
    ctx.ellipse(0, 0, 18, 24, 0, 0, Math.PI * 2);
    ctx.fill();

    // Reset Shadow for sharp finger joints
    ctx.shadowBlur = 0;

    // 5 Distinct Segmented Fingers (Thumb, Index, Middle, Ring, Pinky)
    const fingerLengths = [22, 34, 38, 35, 28];
    const fingerAngles = [-0.65, -0.28, 0, 0.28, 0.58];
    const fingerNames = ["Thumb", "Index", "Middle", "Ring", "Pinky"];

    for (let i = 0; i < 5; i++) {
      let fLen = fingerLengths[i];
      let fAngle = fingerAngles[i];

      // Handshape Adjustments for ISL Signs
      if (handShape === "POINT" && i !== 1) fLen = 14; // Fold non-index fingers
      if (handShape === "FIST") fLen = 16;
      if (handShape === "PULSE" && i > 1) fLen = 15;

      const seg1X = Math.sin(fAngle) * (fLen * 0.5);
      const seg1Y = -Math.cos(fAngle) * (fLen * 0.5);

      const tipX = Math.sin(fAngle) * fLen;
      const tipY = -Math.cos(fAngle) * fLen;

      // Proximal Joint Segment (Base -> Knuckle)
      ctx.strokeStyle = "#f43f5e";
      ctx.lineWidth = 6;
      ctx.lineCap = "round";

      ctx.beginPath();
      ctx.moveTo(0, -8);
      ctx.lineTo(seg1X, seg1Y);
      ctx.stroke();

      // Distal Joint Segment (Knuckle -> Tip)
      ctx.strokeStyle = "#ffe4e6";
      ctx.lineWidth = 5;

      ctx.beginPath();
      ctx.moveTo(seg1X, seg1Y);
      ctx.lineTo(tipX, tipY);
      ctx.stroke();

      // Knuckle Joint Ring (Bright Highlight)
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath();
      ctx.arc(seg1X, seg1Y, 3, 0, Math.PI * 2);
      ctx.fill();

      // Fingertip Node (Fingernail Glow)
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(tipX, tipY, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }

  function renderSubwaySurfersBoyAvatar() {
    if (!actx) return;
    actx.clearRect(0, 0, avatarCanvas.width, avatarCanvas.height);

    const time = Date.now() * 0.003;
    const cx = avatarCanvas.width / 2;
    const cy = avatarCanvas.height / 2 + 15;

    // Cycle Gloss Sequence
    signCycleTimer++;
    if (signCycleTimer % 100 === 0 && activeGlosses.length > 0) {
      currentGlossIndex = (currentGlossIndex + 1) % activeGlosses.length;
      if (nowSigningTag) {
        nowSigningTag.innerText = `Now Signing: [${activeGlosses[currentGlossIndex]}]`;
      }
    }

    const currentGloss = activeGlosses[currentGlossIndex] || "DOCTOR";

    // Subway Surfers Energetic Bounce Motion
    const bounceY = Math.sin(time * 3) * 4;
    const swayX = Math.cos(time * 1.5) * 3;

    // 1. STYLIZED SUBWAY SURFERS ENVIRONMENT
    const bgGrad = actx.createRadialGradient(cx, cy - 80, 40, cx, cy, 320);
    bgGrad.addColorStop(0, "#1e1b4b");
    bgGrad.addColorStop(0.6, "#0f172a");
    bgGrad.addColorStop(1, "#020617");
    actx.fillStyle = bgGrad;
    actx.fillRect(0, 0, avatarCanvas.width, avatarCanvas.height);

    // 2. SUBWAY BOY TORSO & DENIM JACKET / HOODIE
    const shoulderY = cy - 25 + bounceY;
    const shoulderL_X = cx - 90 + swayX;
    const shoulderR_X = cx + 90 + swayX;

    // Denim Jacket Body
    const jacketGrad = actx.createLinearGradient(cx - 75, shoulderY, cx + 75, cy + 140);
    jacketGrad.addColorStop(0, "#0284c7"); // Subway Surfers Blue Denim
    jacketGrad.addColorStop(0.5, "#0369a1");
    jacketGrad.addColorStop(1, "#0f172a");

    actx.fillStyle = jacketGrad;
    actx.beginPath();
    actx.moveTo(shoulderL_X, shoulderY);
    actx.lineTo(shoulderR_X, shoulderY);
    actx.lineTo(cx + 70, cy + 140);
    actx.lineTo(cx - 70, cy + 140);
    actx.closePath();
    actx.fill();

    // Red Hoodie Collar & Zipper
    actx.strokeStyle = "#ef4444"; // Red accent
    actx.lineWidth = 5;
    actx.beginPath();
    actx.moveTo(cx - 25, shoulderY);
    actx.lineTo(cx, shoulderY + 45);
    actx.lineTo(cx + 25, shoulderY);
    actx.stroke();

    // Zipper line
    actx.strokeStyle = "#fbbf24";
    actx.lineWidth = 2;
    actx.beginPath();
    actx.moveTo(cx, shoulderY + 45);
    actx.lineTo(cx, cy + 140);
    actx.stroke();

    // 3. SUBWAY SURFERS BOY HEAD & BACKWARD CAP (Jake Style)
    const headX = cx + swayX;
    const headY = cy - 115 + bounceY;

    // Neck
    actx.fillStyle = "#ffe4e6";
    actx.fillRect(headX - 16, headY + 32, 32, 25);

    // Face Oval
    actx.fillStyle = "#ffe4e6";
    actx.beginPath();
    actx.ellipse(headX, headY, 46, 54, 0, 0, Math.PI * 2);
    actx.fill();
    actx.strokeStyle = "#fb7185";
    actx.lineWidth = 2;
    actx.stroke();

    // Backward Cap (Subway Surfers Iconic Red/Yellow Cap)
    actx.fillStyle = "#dc2626"; // Cap Red
    actx.beginPath();
    actx.arc(headX, headY - 12, 49, Math.PI, Math.PI * 2);
    actx.fill();

    // Cap Visor (Pointing Backward)
    actx.fillStyle = "#fbbf24"; // Cap Yellow Visor
    actx.beginPath();
    actx.ellipse(headX + 25, headY - 18, 30, 8, 0.2, 0, Math.PI * 2);
    actx.fill();

    // Cool Hair Tuft under Cap
    actx.fillStyle = "#f59e0b";
    actx.beginPath();
    actx.arc(headX - 25, headY - 22, 12, 0, Math.PI * 2);
    actx.fill();

    // Eyes (Subway Surfers Big Expressive Anime Eyes)
    const blink = Math.sin(time * 0.4) > 0.95 ? 0.1 : 1;
    actx.fillStyle = "#ffffff";
    actx.beginPath(); actx.ellipse(headX - 18, headY - 6, 10, 8 * blink, 0, 0, Math.PI * 2); actx.fill();
    actx.beginPath(); actx.ellipse(headX + 18, headY - 6, 10, 8 * blink, 0, 0, Math.PI * 2); actx.fill();

    // Blue Pupils with Catchlight Highlights
    actx.fillStyle = "#0284c7";
    actx.beginPath(); actx.arc(headX - 18, headY - 6, 5 * blink, 0, Math.PI * 2); actx.fill();
    actx.beginPath(); actx.arc(headX + 18, headY - 6, 5 * blink, 0, Math.PI * 2); actx.fill();

    actx.fillStyle = "#ffffff";
    actx.beginPath(); actx.arc(headX - 16, headY - 8, 2 * blink, 0, Math.PI * 2); actx.fill();
    actx.beginPath(); actx.arc(headX + 20, headY - 8, 2 * blink, 0, Math.PI * 2); actx.fill();

    // Eyebrows (Dynamic NMM)
    const browY = activeNMM.eyebrows === "RAISED" ? -18 : -14;
    actx.strokeStyle = "#92400e";
    actx.lineWidth = 4;
    actx.beginPath();
    actx.moveTo(headX - 28, headY + browY);
    actx.lineTo(headX - 8, headY + browY - 2);
    actx.moveTo(headX + 8, headY + browY - 2);
    actx.lineTo(headX + 28, headY + browY);
    actx.stroke();

    // Cool Smile
    actx.strokeStyle = "#e11d48";
    actx.lineWidth = 3;
    actx.beginPath();
    actx.arc(headX, headY + 16, 14, 0.1, Math.PI - 0.1);
    actx.stroke();

    // 4. ISL SIGN LANGUAGE HAND KINEMATICS & ULTRA-CLEAR 3D FINGERS
    let armLeft = { elbow: { x: cx - 110, y: cy + 30 }, wrist: { x: cx - 70, y: cy + 10 }, handShape: "OPEN" };
    let armRight = { elbow: { x: cx + 110, y: cy + 30 }, wrist: { x: cx + 70, y: cy + 10 }, handShape: "OPEN" };

    if (currentGloss === "DOCTOR") {
      armLeft = { elbow: { x: cx - 95, y: cy + 45 }, wrist: { x: cx - 35, y: cy + 35 }, handShape: "OPEN" };
      armRight = { elbow: { x: cx + 65, y: cy + 55 }, wrist: { x: cx - 30 + Math.sin(time * 8) * 10, y: cy + 30 }, handShape: "POINT" };
    } else if (currentGloss === "TRAIN") {
      armLeft = { elbow: { x: cx - 80, y: cy + 45 }, wrist: { x: cx - 30, y: cy + 20 + Math.sin(time * 6) * 18 }, handShape: "POINT" };
      armRight = { elbow: { x: cx + 80, y: cy + 45 }, wrist: { x: cx + 30, y: cy + 20 - Math.sin(time * 6) * 18 }, handShape: "POINT" };
    } else if (currentGloss === "STATION" || currentGloss === "HOME") {
      armLeft = { elbow: { x: cx - 95, y: cy + 35 }, wrist: { x: cx - 20, y: cy - 25 }, handShape: "OPEN" };
      armRight = { elbow: { x: cx + 95, y: cy + 35 }, wrist: { x: cx + 20, y: cy - 25 }, handShape: "OPEN" };
    } else if (currentGloss === "GO" || currentGloss === "TOMORROW") {
      armLeft = { elbow: { x: cx - 105, y: cy + 45 }, wrist: { x: cx - 65, y: cy + 25 }, handShape: "OPEN" };
      armRight = { elbow: { x: cx + 105, y: cy + 25 }, wrist: { x: cx + 85 + Math.sin(time * 4) * 25, y: cy - 35 }, handShape: "POINT" };
    }

    // Draw Denim Sleeves & Arms
    actx.strokeStyle = "#ffe4e6";
    actx.lineWidth = 16;
    actx.lineCap = "round";
    actx.lineJoin = "round";

    // Left Arm
    actx.beginPath();
    actx.moveTo(shoulderL_X, shoulderY);
    actx.lineTo(armLeft.elbow.x, armLeft.elbow.y);
    actx.lineTo(armLeft.wrist.x, armLeft.wrist.y);
    actx.stroke();

    // Draw Enlarged Clear 3D Fingers on Left Hand
    drawClearSubwaySurfersHand(actx, armLeft.wrist.x, armLeft.wrist.y, -0.4, armLeft.handShape);

    // Right Arm
    actx.beginPath();
    actx.moveTo(shoulderR_X, shoulderY);
    actx.lineTo(armRight.elbow.x, armRight.elbow.y);
    actx.lineTo(armRight.wrist.x, armRight.wrist.y);
    actx.stroke();

    // Draw Enlarged Clear 3D Fingers on Right Hand
    drawClearSubwaySurfersHand(actx, armRight.wrist.x, armRight.wrist.y, 0.4, armRight.handShape);

    if (isAvatarPlaying) {
      requestAnimationFrame(renderSubwaySurfersBoyAvatar);
    }
  }

  renderSubwaySurfersBoyAvatar();

  // Play/Pause Avatar Controls
  const btnPlayAvatar = document.getElementById("btnPlayAvatar");
  const btnPauseAvatar = document.getElementById("btnPauseAvatar");

  if (btnPlayAvatar && btnPauseAvatar) {
    btnPlayAvatar.addEventListener("click", () => {
      if (!isAvatarPlaying) {
        isAvatarPlaying = true;
        renderSubwaySurfersBoyAvatar();
      }
    });
    btnPauseAvatar.addEventListener("click", () => {
      isAvatarPlaying = false;
    });
  }

  // Landmark Canvas Simulation for Live Camera Tab
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

  // Text-to-ISL Input in Chat Tab
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
            <div class="bubble-trans">Subway Surfers 3D Boy Avatar performing sign language live.</div>
          `;
          chatHistory.appendChild(agentBubble);

          activeGlosses = data.isl_gloss_sequence;
          activeNMM = data.nmm_specifications;
          currentGlossIndex = 0;
        }
      } catch (err) {}

      userTextInput.value = "";
      chatHistory.scrollTop = chatHistory.scrollHeight;
    });
  }
});
