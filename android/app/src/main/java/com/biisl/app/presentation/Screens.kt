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

    val frameAnalyzer = androidx.compose.runtime.remember {
        com.biisl.app.camera.FrameAnalyzer(frameSamplingRateMs = 100L) { image, rotation, ts ->
            // In a real implementation, this feeds to InferenceEngine
            // Log.d("CameraScreen", "Frame processed: $ts")
        }
    }

    androidx.compose.runtime.DisposableEffect(Unit) {
        onDispose {
            cameraPipeline.shutdown()
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
    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("Research Diagnostics", style = MaterialTheme.typography.headlineMedium)
        Text("Performance Metrics")
        Text("Inference Time: -- ms")
        Text("Active NMM Tags: None")
    }
}
