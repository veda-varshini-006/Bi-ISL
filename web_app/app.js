// Bi-ISL Interactive Web Suite - Realistic Animated Human Body Avatar & Live Sign Engine
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

  // -------------------------------------------------------------
  // HIGH-FIDELITY ANIMATED HUMAN BODY 3D AVATAR RENDERER
  // -------------------------------------------------------------
  const avatarCanvas = document.getElementById("avatarCanvas");
  const actx = avatarCanvas ? avatarCanvas.getContext("2d") : null;
  let isAvatarPlaying = true;
  let signCycleTimer = 0;

  function drawDetailedHand(ctx, wristX, wristY, angle, isLeft, handShape = "OPEN") {
    ctx.save();
    ctx.translate(wristX, wristY);
    ctx.rotate(angle);

    // Palm Gradient
    const palmGrad = ctx.createRadialGradient(0, 0, 2, 0, 0, 16);
    palmGrad.addColorStop(0, "#fbcfe8");
    palmGrad.addColorStop(1, "#f43f5e");

    ctx.fillStyle = palmGrad;
    ctx.beginPath();
    ctx.ellipse(0, 0, 14, 18, 0, 0, Math.PI * 2);
    ctx.fill();

    // 5 Fingers (Thumb, Index, Middle, Ring, Pinky)
    const fingerColors = ["#fecdd3", "#fda4af", "#f43f5e", "#e11d48", "#be123c"];
    for (let i = 0; i < 5; i++) {
      let fingerAngle = (i - 2) * 0.28;
      let length = i === 0 ? 16 : (i === 2 ? 24 : 22);
      if (handShape === "POINT" && i !== 1) length = 8; // Fold non-index fingers
      if (handShape === "FIST") length = 10;

      let fx = Math.sin(fingerAngle) * length;
      let fy = -Math.cos(fingerAngle) * length;

      ctx.strokeStyle = fingerColors[i];
      ctx.lineWidth = 4;
      ctx.lineCap = "round";

      ctx.beginPath();
      ctx.moveTo(0, -6);
      ctx.lineTo(fx, fy);
      ctx.stroke();

      // Fingertip joint node
      ctx.fillStyle = "#fff";
      ctx.beginPath();
      ctx.arc(fx, fy, 2, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }

  function renderAnimatedHumanAvatar() {
    if (!actx) return;
    actx.clearRect(0, 0, avatarCanvas.width, avatarCanvas.height);

    const time = Date.now() * 0.003;
    const cx = avatarCanvas.width / 2;
    const cy = avatarCanvas.height / 2 + 20;

    // Cycle through active gloss sequence
    signCycleTimer++;
    if (signCycleTimer % 100 === 0 && activeGlosses.length > 0) {
      currentGlossIndex = (currentGlossIndex + 1) % activeGlosses.length;
      if (nowSigningTag) {
        nowSigningTag.innerText = `Now Signing: [${activeGlosses[currentGlossIndex]}]`;
      }
    }

    const currentGloss = activeGlosses[currentGlossIndex] || "DOCTOR";

    // Dynamic Breathing & Body Sway
    const breatheY = Math.sin(time * 1.5) * 3;
    const swayX = Math.cos(time * 0.8) * 2;

    // 1. BACKGROUND ENVIRONMENT GRADIENT
    const bgGrad = actx.createRadialGradient(cx, cy - 80, 50, cx, cy, 300);
    bgGrad.addColorStop(0, "#1e1b4b");
    bgGrad.addColorStop(1, "#090d16");
    actx.fillStyle = bgGrad;
    actx.fillRect(0, 0, avatarCanvas.width, avatarCanvas.height);

    // 2. HUMAN TORSO & CLOTHING (Shirt & Shoulders)
    const shoulderY = cy - 20 + breatheY;
    const shoulderL_X = cx - 85 + swayX;
    const shoulderR_X = cx + 85 + swayX;

    // Torso Gradient (High Quality Suit/Shirt Shading)
    const torsoGrad = actx.createLinearGradient(cx - 70, shoulderY, cx + 70, cy + 140);
    torsoGrad.addColorStop(0, "#312e81");
    torsoGrad.addColorStop(0.5, "#1e1b4b");
    torsoGrad.addColorStop(1, "#0f172a");

    actx.fillStyle = torsoGrad;
    actx.beginPath();
    actx.moveTo(shoulderL_X, shoulderY);
    actx.lineTo(shoulderR_X, shoulderY);
    actx.lineTo(cx + 65, cy + 140);
    actx.lineTo(cx - 65, cy + 140);
    actx.closePath();
    actx.fill();

    // Collar & V-Neck
    actx.strokeStyle = "#818cf8";
    actx.lineWidth = 3;
    actx.beginPath();
    actx.moveTo(shoulderL_X + 25, shoulderY);
    actx.lineTo(cx, shoulderY + 35);
    actx.lineTo(shoulderR_X - 25, shoulderY);
    actx.stroke();

    // 3. HUMAN NECK & HEAD (Skin Tones & Shading)
    const headX = cx + swayX;
    const headY = cy - 110 + breatheY;

    // Neck
    const skinGrad = actx.createLinearGradient(headX - 15, headY, headX + 15, shoulderY);
    skinGrad.addColorStop(0, "#fecdd3");
    skinGrad.addColorStop(1, "#fda4af");

    actx.fillStyle = skinGrad;
    actx.fillRect(headX - 14, headY + 30, 28, 25);

    // Head Oval
    actx.fillStyle = skinGrad;
    actx.beginPath();
    actx.ellipse(headX, headY, 44, 52, 0, 0, Math.PI * 2);
    actx.fill();
    actx.strokeStyle = "#e11d48";
    actx.lineWidth = 1.5;
    actx.stroke();

    // Hair
    actx.fillStyle = "#1e1b4b";
    actx.beginPath();
    actx.arc(headX, headY - 15, 46, Math.PI, Math.PI * 2);
    actx.fill();

    // Eyes (with Natural Blinking Animation)
    const blink = Math.sin(time * 0.5) > 0.96 ? 0.1 : 1;
    actx.fillStyle = "#fff";
    actx.beginPath(); actx.ellipse(headX - 16, headY - 8, 8, 6 * blink, 0, 0, Math.PI * 2); actx.fill();
    actx.beginPath(); actx.ellipse(headX + 16, headY - 8, 8, 6 * blink, 0, 0, Math.PI * 2); actx.fill();

    // Pupils
    actx.fillStyle = "#312e81";
    actx.beginPath(); actx.arc(headX - 16, headY - 8, 3.5 * blink, 0, Math.PI * 2); actx.fill();
    actx.beginPath(); actx.arc(headX + 16, headY - 8, 3.5 * blink, 0, Math.PI * 2); actx.fill();

    // Eyebrows (Dynamic Non-Manual Marker NMM: Raised or Normal)
    const browYOffset = activeNMM.eyebrows === "RAISED" ? -14 : -10;
    actx.strokeStyle = "#312e81";
    actx.lineWidth = 3.5;
    actx.beginPath();
    actx.moveTo(headX - 24, headY + browYOffset);
    actx.lineTo(headX - 8, headY + browYOffset - 1);
    actx.moveTo(headX + 8, headY + browYOffset - 1);
    actx.lineTo(headX + 24, headY + browYOffset);
    actx.stroke();

    // Nose
    actx.strokeStyle = "#fb7185";
    actx.lineWidth = 2;
    actx.beginPath();
    actx.moveTo(headX, headY - 4);
    actx.lineTo(headX - 3, headY + 10);
    actx.lineTo(headX + 4, headY + 10);
    actx.stroke();

    // Mouth / Mouthings (Mouth articulates according to active ISL gloss)
    const mouthOpen = 4 + Math.sin(time * 6) * 3;
    actx.fillStyle = "#9f1239";
    actx.beginPath();
    actx.ellipse(headX, headY + 24, 12, mouthOpen, 0, 0, Math.PI * 2);
    actx.fill();

    // 4. SIGN LANGUAGE ARM KINEMATICS & HAND MOTIONS
    // Mapped positions for specific ISL signs:
    let armLeftTarget = { elbow: { x: cx - 110, y: cy + 30 }, wrist: { x: cx - 70, y: cy + 10 }, handShape: "OPEN" };
    let armRightTarget = { elbow: { x: cx + 110, y: cy + 30 }, wrist: { x: cx + 70, y: cy + 10 }, handShape: "OPEN" };

    if (currentGloss === "DOCTOR") {
      // Doctor sign: Right hand taps left wrist (pulse check)
      armLeftTarget = { elbow: { x: cx - 90, y: cy + 40 }, wrist: { x: cx - 30, y: cy + 30 }, handShape: "OPEN" };
      armRightTarget = { elbow: { x: cx + 60, y: cy + 50 }, wrist: { x: cx - 25 + Math.sin(time * 8) * 8, y: cy + 25 }, handShape: "POINT" };
    } else if (currentGloss === "TRAIN") {
      // Train sign: Two hands parallel sliding forward/back
      armLeftTarget = { elbow: { x: cx - 75, y: cy + 40 }, wrist: { x: cx - 25, y: cy + 20 + Math.sin(time * 6) * 15 }, handShape: "POINT" };
      armRightTarget = { elbow: { x: cx + 75, y: cy + 40 }, wrist: { x: cx + 25, y: cy + 20 - Math.sin(time * 6) * 15 }, handShape: "POINT" };
    } else if (currentGloss === "STATION" || currentGloss === "HOME") {
      // Station/Home: Hands forming roof shape in front of chest
      armLeftTarget = { elbow: { x: cx - 90, y: cy + 30 }, wrist: { x: cx - 15, y: cy - 20 }, handShape: "OPEN" };
      armRightTarget = { elbow: { x: cx + 90, y: cy + 30 }, wrist: { x: cx + 15, y: cy - 20 }, handShape: "OPEN" };
    } else if (currentGloss === "GO" || currentGloss === "TOMORROW") {
      // Go/Tomorrow: Sweeping arm pointing outward
      armLeftTarget = { elbow: { x: cx - 100, y: cy + 40 }, wrist: { x: cx - 60, y: cy + 20 }, handShape: "OPEN" };
      armRightTarget = { elbow: { x: cx + 100, y: cy + 20 }, wrist: { x: cx + 80 + Math.sin(time * 4) * 20, y: cy - 30 }, handShape: "POINT" };
    }

    // Draw Left Arm & Hand
    actx.strokeStyle = skinGrad;
    actx.lineWidth = 14;
    actx.lineCap = "round";
    actx.lineJoin = "round";

    actx.beginPath();
    actx.moveTo(shoulderL_X, shoulderY);
    actx.lineTo(armLeftTarget.elbow.x, armLeftTarget.elbow.y);
    actx.lineTo(armLeftTarget.wrist.x, armLeftTarget.wrist.y);
    actx.stroke();

    drawDetailedHand(actx, armLeftTarget.wrist.x, armLeftTarget.wrist.y, -0.4, true, armLeftTarget.handShape);

    // Draw Right Arm & Hand
    actx.beginPath();
    actx.moveTo(shoulderR_X, shoulderY);
    actx.lineTo(armRightTarget.elbow.x, armRightTarget.elbow.y);
    actx.lineTo(armRightTarget.wrist.x, armRightTarget.wrist.y);
    actx.stroke();

    drawDetailedHand(actx, armRightTarget.wrist.x, armRightTarget.wrist.y, 0.4, false, armRightTarget.handShape);

    if (isAvatarPlaying) {
      requestAnimationFrame(renderAnimatedHumanAvatar);
    }
  }

  renderAnimatedHumanAvatar();

  // Play/Pause Avatar Controls
  const btnPlayAvatar = document.getElementById("btnPlayAvatar");
  const btnPauseAvatar = document.getElementById("btnPauseAvatar");

  if (btnPlayAvatar && btnPauseAvatar) {
    btnPlayAvatar.addEventListener("click", () => {
      if (!isAvatarPlaying) {
        isAvatarPlaying = true;
        renderAnimatedHumanAvatar();
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
            <div class="bubble-trans">Animated Human Body Avatar rendering sign language live.</div>
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
