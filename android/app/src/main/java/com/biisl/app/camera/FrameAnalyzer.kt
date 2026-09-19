package com.biisl.app.camera

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import android.util.Log

class FrameAnalyzer(
    private val frameSamplingRateMs: Long = 100L, // Sample roughly 10fps
    private val onFrameProcessed: (image: ImageProxy, rotationDegrees: Int, timestamp: Long) -> Unit
) : ImageAnalysis.Analyzer {

    private var lastAnalyzedTimestamp = 0L

    @androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class)
    override fun analyze(image: ImageProxy) {
        val currentTimestamp = System.currentTimeMillis()
        
        // Sampling backpressure logic: skip frames if they arrive too fast
        if (currentTimestamp - lastAnalyzedTimestamp >= frameSamplingRateMs) {
            val rotationDegrees = image.imageInfo.rotationDegrees
            val timestamp = image.imageInfo.timestamp // Hardware timestamp

            // Log details for research diagnostics
            Log.d("FrameAnalyzer", "Processing frame: rotation=$rotationDegrees, ts=$timestamp")

            // Pass to the callback (which might route to InferenceEngine)
            onFrameProcessed(image, rotationDegrees, timestamp)
            
            lastAnalyzedTimestamp = currentTimestamp
        }
        
        // ALWAYS close the imageProxy to prevent stalling the CameraX pipeline
        image.close()
    }
}
