package com.biisl.app.presentation

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController

@Composable
fun AppNavigation() {
    val navController = rememberNavController()
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

@Composable
fun ConversationScreen(navController: NavController) {
    Column(modifier = Modifier.fillMaxSize().padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
        Text("Conversation Screen", style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(16.dp))
        Button(onClick = { navController.navigate("camera") }) { Text("Open Camera (ISL to Eng)") }
        Button(onClick = { navController.navigate("input") }) { Text("Text/Speech Input (Eng to ISL)") }
        Button(onClick = { navController.navigate("settings") }) { Text("Settings") }
        Button(onClick = { navController.navigate("diagnostics") }) { Text("Diagnostics") }
    }
}

@Composable
fun CameraScreen(navController: NavController) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current

    val cameraPipeline = androidx.compose.runtime.remember {
        com.biisl.app.camera.CameraPipeline(context)
    }

    val executorchEngine = androidx.compose.runtime.remember {
        com.biisl.app.inference.ExecuTorchInferenceEngine(context).apply {
            // Load the E8 baseline model (assuming .pte format)
            initialize("baseline_best_v1.pte")
        }
    }

    val resultAggregator = androidx.compose.runtime.remember {
        com.biisl.app.mediapipe.ResultAggregator { combinedResult ->
            // Pass the combined result to InferenceEngine
            // Here we would flatten the landmarks to a FloatArray
            val dummyLandmarks = FloatArray(100) 
            val translation = executorchEngine.runInferenceOnLandmarks(dummyLandmarks)
            android.util.Log.d("CameraScreen", "Got combined landmarks, latency: ${combinedResult.preprocessingLatencyMs}ms, Translation: $translation")
        }
    }

    val mediaPipeHelper = androidx.compose.runtime.remember {
        com.biisl.app.mediapipe.MediaPipeHelper(context, resultAggregator)
    }

    val frameAnalyzer = androidx.compose.runtime.remember {
        com.biisl.app.camera.FrameAnalyzer(frameSamplingRateMs = 100L) { image, rotation, ts ->
            mediaPipeHelper.detectLiveStream(image, isFrontCamera = false)
        }
    }

    androidx.compose.runtime.DisposableEffect(Unit) {
        onDispose {
            cameraPipeline.shutdown()
            mediaPipeHelper.shutdown()
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        androidx.compose.ui.viewinterop.AndroidView(
            factory = { ctx ->
                androidx.camera.view.PreviewView(ctx).apply {
                    this.scaleType = androidx.camera.view.PreviewView.ScaleType.FILL_CENTER
                    cameraPipeline.bindToLifecycle(lifecycleOwner, this, frameAnalyzer)
                }
            },
            modifier = Modifier.fillMaxSize()
        )

        Button(
            onClick = { navController.navigate("translation") },
            modifier = Modifier.align(Alignment.BottomCenter).padding(32.dp)
        ) {
            Text("Simulate Translation")
        }
    }
}

@Composable
fun InputScreen(navController: NavController) {
    Column(modifier = Modifier.fillMaxSize().padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
        Text("Speech/Text Input Screen")
        OutlinedTextField(value = "", onValueChange = {}, label = { Text("Enter English text") })
        Spacer(modifier = Modifier.height(16.dp))
        Button(onClick = { navController.navigate("avatar") }) { Text("Translate to ISL Avatar") }
    }
}

@Composable
fun TranslationResultScreen(navController: NavController) {
    Column(modifier = Modifier.fillMaxSize().padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
        Text("Translation Result")
        Text("English Output: [Simulated output from InferenceEngine]", style = MaterialTheme.typography.bodyLarge)
        Spacer(modifier = Modifier.height(16.dp))
        Button(onClick = { navController.navigate("conversation") }) { Text("Back to Conversation") }
    }
}

@Composable
fun AvatarViewScreen(navController: NavController) {
    Column(modifier = Modifier.fillMaxSize().padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
        Text("3D Avatar View")
        Text("Rendering ISL Representation via AvatarRenderer", style = MaterialTheme.typography.bodyLarge)
        Spacer(modifier = Modifier.height(16.dp))
        Button(onClick = { navController.navigate("conversation") }) { Text("Back to Conversation") }
    }
}

@Composable
fun SettingsScreen(navController: NavController) {
    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("Settings", style = MaterialTheme.typography.headlineMedium)
        Text("Model Path Configuration")
        Text("Avatar Selection")
    }
}

@Composable
fun ResearchDiagnosticsScreen(navController: NavController) {
    // In a real app we'd pass the engine instance or hoist the state.
    // For demonstration of the shell, we display placeholders that map to the BenchmarkStats.
    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("Research Diagnostics (E8 Baseline)", style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(16.dp))
        Text("Performance Metrics", style = MaterialTheme.typography.titleLarge)
        
        Card(modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Model: baseline_best_v1.pte (XNNPACK)")
                Text("Load Time: 420 ms")
                Text("Warm-up Time: 15 ms")
                Text("p50 Latency: 8 ms")
                Text("p95 Latency: 12 ms")
                Text("Memory Usage: 45.2 MB")
                Text("Model Size: 12.4 MB")
            }
        }
        
        Text("Active NMM Tags: None")
        Spacer(modifier = Modifier.height(16.dp))
        Button(onClick = { navController.navigate("conversation") }) { Text("Back") }
    }
}
