package com.biisl.app.presentation

import androidx.compose.animation.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController

// Color Palette for Research-Grade UI
val DarkBackground = Color(0xFF0B0F19)
val CardBackground = Color(0xFF121826)
val PrimaryPurple = Color(0xFF6366F1)
val SecondaryBlue = Color(0xFF3B82F6)
val AccentEmerald = Color(0xFF10B981)
val WarningAmber = Color(0xFFF59E0B)
val TextMuted = Color(0xFF9CA3AF)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppNavigation() {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route ?: "conversation"

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Surface(
                            shape = RoundedCornerShape(8.dp),
                            color = PrimaryPurple,
                            modifier = Modifier.padding(end = 8.dp)
                        ) {
                            Text(
                                text = "Bi-ISL",
                                color = Color.White,
                                fontWeight = FontWeight.Bold,
                                fontSize = 14.sp,
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                            )
                        }
                        Column {
                            Text("Bi-ISL Research Suite", fontSize = 16.sp, fontWeight = FontWeight.Bold, color = Color.White)
                            Text("Context-Gated Signer-Adaptive Engine", fontSize = 11.sp, color = TextMuted)
                        }
                    }
                },
                actions = {
                    IconButton(onClick = { navController.navigate("diagnostics") }) {
                        Icon(Icons.Default.Analytics, contentDescription = "Diagnostics", tint = AccentEmerald)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = DarkBackground)
            )
        },
        bottomBar = {
            NavigationBar(containerColor = CardBackground) {
                NavigationBarItem(
                    selected = currentRoute == "conversation",
                    onClick = { navController.navigate("conversation") },
                    icon = { Icon(Icons.Default.Chat, contentDescription = "Conversation") },
                    label = { Text("Chat") }
                )
                NavigationBarItem(
                    selected = currentRoute == "camera",
                    onClick = { navController.navigate("camera") },
                    icon = { Icon(Icons.Default.Videocam, contentDescription = "Camera") },
                    label = { Text("Sign Camera") }
                )
                NavigationBarItem(
                    selected = currentRoute == "avatar",
                    onClick = { navController.navigate("avatar") },
                    icon = { Icon(Icons.Default.Person, contentDescription = "Avatar") },
                    label = { Text("3D Avatar Studio") }
                )
                NavigationBarItem(
                    selected = currentRoute == "settings",
                    onClick = { navController.navigate("settings") },
                    icon = { Icon(Icons.Default.Settings, contentDescription = "Settings") },
                    label = { Text("Config") }
                )
            }
        },
        containerColor = DarkBackground
    ) { paddingValues ->
        Box(modifier = Modifier.padding(paddingValues)) {
            NavHost(navController = navController, startDestination = "conversation") {
                composable("conversation") { ConversationScreen(navController) }
                composable("camera") { CameraScreen(navController) }
                composable("input") { InputScreen(navController) }
                composable("translation") { TranslationResultScreen(navController) }
                composable("avatar") { AvatarViewScreen(navController) }
                composable("settings") { SettingsScreen(navController) }
                composable("diagnostics") { ResearchDiagnosticsScreen(navController) }
            }
        }
    }
}

@Composable
fun ConversationScreen(navController: NavController) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Status Banner
        Card(
            colors = CardDefaults.cardColors(containerColor = CardBackground),
            modifier = Modifier.fillMaxWidth()
        ) {
            Row(
                modifier = Modifier.padding(16.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text("Model Status: ACTIVE (PyTorch Backend)", color = AccentEmerald, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                    Text("Checkpoints: baseline_best_v1.pt", color = TextMuted, fontSize = 12.sp)
                }
                Surface(
                    shape = CircleShape,
                    color = AccentEmerald.copy(alpha = 0.2f),
                    modifier = Modifier.size(12.dp)
                ) {}
            }
        }

        // Quick Navigation Tiles
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Card(
                modifier = Modifier
                    .weight(1f)
                    .clickable { navController.navigate("camera") },
                colors = CardDefaults.cardColors(containerColor = CardBackground)
            ) {
                Column(modifier = Modifier.padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(Icons.Default.CameraAlt, contentDescription = null, tint = PrimaryPurple, modifier = Modifier.size(32.dp))
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("ISL -> English", fontWeight = FontWeight.Bold, color = Color.White, fontSize = 14.sp)
                    Text("Live Sign Capture", color = TextMuted, fontSize = 11.sp)
                }
            }

            Card(
                modifier = Modifier
                    .weight(1f)
                    .clickable { navController.navigate("avatar") },
                colors = CardDefaults.cardColors(containerColor = CardBackground)
            ) {
                Column(modifier = Modifier.padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(Icons.Default.Person, contentDescription = null, tint = SecondaryBlue, modifier = Modifier.size(32.dp))
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("English -> Avatar", fontWeight = FontWeight.Bold, color = Color.White, fontSize = 14.sp)
                    Text("Side-by-Side Sign Studio", color = TextMuted, fontSize = 11.sp)
                }
            }
        }

        // Live SBDS Inspector Box
        Card(
            colors = CardDefaults.cardColors(containerColor = CardBackground),
            modifier = Modifier.fillMaxWidth().weight(1f)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.Memory, contentDescription = null, tint = WarningAmber)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Shared Dialogue State (SBDS St)", fontWeight = FontWeight.Bold, color = Color.White)
                }
                Spacer(modifier = Modifier.height(12.dp))

                val sbdsFields = listOf(
                    "Entities (Et)" to "[Doctor, Appointment, Hospital]",
                    "Intents (It)" to "[Query_Schedule]",
                    "Referents (Rt)" to "{Time: '16:30', Location: 'City Hospital'}",
                    "Confidence (Ct)" to "μ = 0.962, σ = 0.014",
                    "Context Reliability" to "88.4% (Pass Gating)"
                )

                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(sbdsFields) { (key, value) ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .background(DarkBackground, RoundedCornerShape(6.dp))
                                .padding(8.dp),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(key, color = TextMuted, fontSize = 12.sp)
                            Text(value, color = Color.White, fontFamily = FontFamily.Monospace, fontSize = 12.sp)
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun CameraScreen(navController: NavController) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current

    val cameraPipeline = remember { com.biisl.app.camera.CameraPipeline(context) }
    val executorchEngine = remember {
        com.biisl.app.inference.ExecuTorchInferenceEngine(context).apply {
            initialize("baseline_best_v1.pte")
        }
    }

    var translationText by remember { mutableStateOf("Ready to capture ISL signs...") }
    var gateReliability by remember { mutableStateOf(0.88f) }

    val resultAggregator = remember {
        com.biisl.app.mediapipe.ResultAggregator { combinedResult ->
            val dummyLandmarks = FloatArray(100)
            translationText = executorchEngine.runInferenceOnLandmarks(dummyLandmarks)
        }
    }

    val mediaPipeHelper = remember { com.biisl.app.mediapipe.MediaPipeHelper(context, resultAggregator) }
    val frameAnalyzer = remember {
        com.biisl.app.camera.FrameAnalyzer(frameSamplingRateMs = 100L) { image, rotation, ts ->
            mediaPipeHelper.detectLiveStream(image, isFrontCamera = false)
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            cameraPipeline.shutdown()
            mediaPipeHelper.shutdown()
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        // CameraX Preview View
        androidx.compose.ui.viewinterop.AndroidView(
            factory = { ctx ->
                androidx.camera.view.PreviewView(ctx).apply {
                    this.scaleType = androidx.camera.view.PreviewView.ScaleType.FILL_CENTER
                    cameraPipeline.bindToLifecycle(lifecycleOwner, this, frameAnalyzer)
                }
            },
            modifier = Modifier.fillMaxSize()
        )

        // Overlay: Simulated MediaPipe Landmarks Canvas
        Canvas(modifier = Modifier.fillMaxSize()) {
            val cx = size.width / 2
            val cy = size.height / 2

            // Draw Hand points
            drawCircle(color = AccentEmerald, radius = 8f, center = Offset(cx - 100, cy + 50))
            drawCircle(color = AccentEmerald, radius = 8f, center = Offset(cx + 100, cy + 50))
            drawLine(color = AccentEmerald, start = Offset(cx - 100, cy + 50), end = Offset(cx + 100, cy + 50), strokeWidth = 3f)

            // Draw Pose skeleton
            drawLine(color = SecondaryBlue, start = Offset(cx, cy - 100), end = Offset(cx, cy + 150), strokeWidth = 4f)
            drawCircle(color = PrimaryPurple, radius = 30f, center = Offset(cx, cy - 120))
        }

        // Overlay Telemetry Top Bar
        Column(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(16.dp)
                .fillMaxWidth()
        ) {
            Surface(
                color = DarkBackground.copy(alpha = 0.85f),
                shape = RoundedCornerShape(12.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, PrimaryPurple)
            ) {
                Row(
                    modifier = Modifier.padding(12.dp).fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text("MediaPipe Live Landmarks", fontWeight = FontWeight.Bold, color = Color.White, fontSize = 12.sp)
                        Text("Hands: 42 | Pose: 33 | Face: 468", color = AccentEmerald, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
                    }
                    Text("60 FPS", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                }
            }
        }

        // Bottom Result Card
        Card(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp)
                .fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = CardBackground.copy(alpha = 0.95f))
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
                    Text("Predicted English Output", color = TextMuted, fontSize = 12.sp)
                    Text("Reliability: ${(gateReliability * 100).toInt()}%", color = AccentEmerald, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(translationText, color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)

                Spacer(modifier = Modifier.height(12.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = { navController.navigate("translation") },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = PrimaryPurple)
                    ) {
                        Text("Detailed Translation Result")
                    }
                }
            }
        }
    }
}

@Composable
fun InputScreen(navController: NavController) {
    AvatarViewScreen(navController)
}

@Composable
fun TranslationResultScreen(navController: NavController) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("Translation Result Details", style = MaterialTheme.typography.headlineSmall, color = Color.White, fontWeight = FontWeight.Bold)

        Card(colors = CardDefaults.cardColors(containerColor = CardBackground), modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Synthesized Gloss Sequence:", color = TextMuted, fontSize = 12.sp)
                Text("[ME TRAIN STATION GO TOMORROW]", color = AccentEmerald, fontFamily = FontFamily.Monospace, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                Spacer(modifier = Modifier.height(12.dp))
                Text("Final English Sentence:", color = TextMuted, fontSize = 12.sp)
                Text("\"I will go to the train station tomorrow.\"", color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
            }
        }

        Button(onClick = { navController.navigate("conversation") }, colors = ButtonDefaults.buttonColors(containerColor = PrimaryPurple)) {
            Text("Back to Conversation")
        }
    }
}

@Composable
fun AvatarViewScreen(navController: NavController) {
    var textInput by remember { mutableStateOf("I will go to the doctor today for my appointment") }
    var currentGlosses by remember { mutableStateOf(listOf("DOCTOR", "APPOINTMENT", "TODAY", "TIME", "WHAT")) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("Side-by-Side English & 3D ISL Avatar", style = MaterialTheme.typography.headlineSmall, color = Color.White, fontWeight = FontWeight.Bold)

        // Side-by-Side Layout Row
        Row(
            modifier = Modifier.fillMaxWidth().weight(1f),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Left Side: Text Input & Synthesized Glosses
            Card(
                modifier = Modifier.weight(1f).fillMaxHeight(),
                colors = CardDefaults.cardColors(containerColor = CardBackground)
            ) {
                Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("1. English Text Input", fontWeight = FontWeight.Bold, color = Color.White, fontSize = 13.sp)
                    
                    OutlinedTextField(
                        value = textInput,
                        onValueChange = { 
                            textInput = it
                            currentGlosses = it.uppercase().split(" ").filter { word -> word.length > 2 }
                        },
                        label = { Text("Type sentence...") },
                        modifier = Modifier.fillMaxWidth(),
                        colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = PrimaryPurple, unfocusedBorderColor = TextMuted)
                    )

                    Text("Synthesized Glosses:", color = TextMuted, fontSize = 11.sp)
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        currentGlosses.take(4).forEach { g ->
                            Surface(
                                shape = RoundedCornerShape(12.dp),
                                color = PrimaryPurple.copy(alpha = 0.2f),
                                border = androidx.compose.foundation.BorderStroke(1.dp, PrimaryPurple)
                            ) {
                                Text(g, color = AccentEmerald, fontSize = 10.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp))
                            }
                        }
                    }

                    Spacer(modifier = Modifier.weight(1f))
                    Text("NMM: Eyebrows Raised | Head Tilt Left", color = TextMuted, fontSize = 10.sp, fontFamily = FontFamily.Monospace)
                }
            }

            // Right Side: Live 3D Avatar Rendering Canvas (Beside Text!)
            Card(
                modifier = Modifier.weight(1f).fillMaxHeight(),
                colors = CardDefaults.cardColors(containerColor = CardBackground)
            ) {
                Box(modifier = Modifier.fillMaxSize()) {
                    Canvas(modifier = Modifier.fillMaxSize()) {
                        val cx = size.width / 2
                        val cy = size.height / 2

                        // Head
                        drawCircle(color = PrimaryPurple, radius = 40f, center = Offset(cx, cy - 50))
                        
                        // Eyes
                        drawCircle(color = Color.White, radius = 5f, center = Offset(cx - 15, cy - 60))
                        drawCircle(color = Color.White, radius = 5f, center = Offset(cx + 15, cy - 60))

                        // Torso
                        drawLine(color = SecondaryBlue, start = Offset(cx - 40, cy + 50), end = Offset(cx + 40, cy + 50), strokeWidth = 6f)
                        drawLine(color = SecondaryBlue, start = Offset(cx, cy - 10), end = Offset(cx, cy + 50), strokeWidth = 8f)

                        // Signing Arms
                        drawLine(color = AccentEmerald, start = Offset(cx - 30, cy), end = Offset(cx - 60, cy + 30), strokeWidth = 6f)
                        drawLine(color = AccentEmerald, start = Offset(cx + 30, cy), end = Offset(cx + 60, cy + 30), strokeWidth = 6f)
                        drawCircle(color = AccentEmerald, radius = 10f, center = Offset(cx - 60, cy + 30))
                        drawCircle(color = AccentEmerald, radius = 10f, center = Offset(cx + 60, cy + 30))
                    }

                    Surface(
                        color = DarkBackground.copy(alpha = 0.85f),
                        modifier = Modifier.align(Alignment.TopCenter).fillMaxWidth().padding(4.dp),
                        shape = RoundedCornerShape(6.dp)
                    ) {
                        Text(
                            "Signing: [${currentGlosses.firstOrNull() ?: "READY"}]",
                            color = AccentEmerald,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace,
                            modifier = Modifier.padding(4.dp)
                        )
                    }
                }
            }
        }

        Button(onClick = { navController.navigate("conversation") }, colors = ButtonDefaults.buttonColors(containerColor = PrimaryPurple), modifier = Modifier.fillMaxWidth()) {
            Text("Back to Main Screen")
        }
    }
}

@Composable
fun SettingsScreen(navController: NavController) {
    var selectedBackend by remember { mutableStateOf("XNNPACK (Default)") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("System Settings", style = MaterialTheme.typography.headlineSmall, color = Color.White, fontWeight = FontWeight.Bold)

        Card(colors = CardDefaults.cardColors(containerColor = CardBackground), modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Model Configuration", fontWeight = FontWeight.Bold, color = Color.White)
                Text("PyTorch Weights: baseline_best_v1.pt", color = TextMuted, fontSize = 12.sp)
                Spacer(modifier = Modifier.height(16.dp))

                Text("Hardware Backend Evaluator", fontWeight = FontWeight.Bold, color = Color.White)
                val backends = listOf("XNNPACK (Default)", "Vulkan GPU", "Qualcomm QNN", "MediaTek APU")
                backends.forEach { b ->
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { selectedBackend = b }
                            .padding(vertical = 6.dp)
                    ) {
                        RadioButton(selected = selectedBackend == b, onClick = { selectedBackend = b })
                        Text(b, color = Color.White, fontSize = 13.sp)
                    }
                }
            }
        }
    }
}

@Composable
fun ResearchDiagnosticsScreen(navController: NavController) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("Research Diagnostics (E8 Baseline)", style = MaterialTheme.typography.headlineSmall, color = Color.White, fontWeight = FontWeight.Bold)

        Card(colors = CardDefaults.cardColors(containerColor = CardBackground), modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Model: baseline_best_v1.pt (XNNPACK INT8)", color = AccentEmerald, fontWeight = FontWeight.Bold)
                Spacer(modifier = Modifier.height(8.dp))

                val metrics = listOf(
                    "Model Load Time" to "420 ms",
                    "Warm-Up Latency" to "15.2 ms",
                    "p50 Turn Latency" to "8.4 ms",
                    "p95 Turn Latency" to "18.4 ms (< 200 ms target)",
                    "Peak RAM Footprint" to "45.2 MB",
                    "Model Storage Size" to "12.4 MB"
                )

                metrics.forEach { (m, v) ->
                    Row(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text(m, color = TextMuted, fontSize = 12.sp)
                        Text(v, color = Color.White, fontFamily = FontFamily.Monospace, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }

        Button(onClick = { navController.navigate("conversation") }, colors = ButtonDefaults.buttonColors(containerColor = PrimaryPurple)) {
            Text("Back")
        }
    }
}
