package com.biisl.app.mediapipe

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MediaPipeHelper(
    val context: Context,
    val resultAggregator: ResultAggregator
) {

    private var handLandmarker: HandLandmarker? = null
    private var poseLandmarker: PoseLandmarker? = null
    private var faceLandmarker: FaceLandmarker? = null

    // Background thread for converting images
    private val backgroundExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    init {
        setupLandmarkers()
    }

    private fun setupLandmarkers() {
        try {
            // Note: In a real app, these .task files MUST exist in assets/
            val handBaseOptions = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build()
            val handOptions = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(handBaseOptions)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumHands(2)
                .setResultListener { result: HandLandmarkerResult, inputImage: MPImage ->
                    // Log.d("MediaPipeHelper", "Hand result received for ts: ${inputImage.timestamp}")
                }
                .setErrorListener { error ->
                    Log.e("MediaPipeHelper", "Hand Error: ${error.message}")
                }
                .build()
            // handLandmarker = HandLandmarker.createFromOptions(context, handOptions)

            val poseBaseOptions = BaseOptions.builder().setModelAssetPath("pose_landmarker_full.task").build()
            val poseOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(poseBaseOptions)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setResultListener { result: PoseLandmarkerResult, inputImage: MPImage ->
                    // Log.d("MediaPipeHelper", "Pose result received for ts: ${inputImage.timestamp}")
                }
                .setErrorListener { error ->
                    Log.e("MediaPipeHelper", "Pose Error: ${error.message}")
                }
                .build()
            // poseLandmarker = PoseLandmarker.createFromOptions(context, poseOptions)

            val faceBaseOptions = BaseOptions.builder().setModelAssetPath("face_landmarker.task").build()
            val faceOptions = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(faceBaseOptions)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setOutputFaceBlendshapes(true)
                .setResultListener { result: FaceLandmarkerResult, inputImage: MPImage ->
                    // Log.d("MediaPipeHelper", "Face result received for ts: ${inputImage.timestamp}")
                }
                .setErrorListener { error ->
                    Log.e("MediaPipeHelper", "Face Error: ${error.message}")
                }
                .build()
            // faceLandmarker = FaceLandmarker.createFromOptions(context, faceOptions)

        } catch (e: Exception) {
            Log.e("MediaPipeHelper", "Models not found in assets, skipping init. Download them to test.", e)
        }
    }

    fun detectLiveStream(imageProxy: ImageProxy, isFrontCamera: Boolean) {
        val frameTimeMs = SystemClock.uptimeMillis() // Start time for latency measurement

        val bitmapBuffer = Bitmap.createBitmap(
            imageProxy.width,
            imageProxy.height,
            Bitmap.Config.ARGB_8888
        )
        // In a real implementation we convert YUV to ARGB here or use ByteBuffer
        // For demonstration of architecture:
        
        val mpImage = BitmapImageBuilder(bitmapBuffer).build()

        val timestamp = imageProxy.imageInfo.timestamp

        // Register the timestamp in the aggregator
        resultAggregator.registerTimestamp(timestamp, frameTimeMs)

        // Asynchronously detect
        backgroundExecutor.execute {
            try {
                // If models were loaded, we would call:
                // handLandmarker?.detectAsync(mpImage, timestamp)
                // poseLandmarker?.detectAsync(mpImage, timestamp)
                // faceLandmarker?.detectAsync(mpImage, timestamp)

                // Simulated callbacks for architecture setup:
                resultAggregator.onHandResult(timestamp, null)
                resultAggregator.onPoseResult(timestamp, null)
                resultAggregator.onFaceResult(timestamp, null)
            } catch (e: Exception) {
                Log.e("MediaPipeHelper", "Error in detectAsync", e)
            } finally {
                imageProxy.close()
            }
        }
    }

    fun shutdown() {
        handLandmarker?.close()
        poseLandmarker?.close()
        faceLandmarker?.close()
        backgroundExecutor.shutdown()
    }
}
