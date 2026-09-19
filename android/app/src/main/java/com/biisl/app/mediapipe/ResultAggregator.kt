package com.biisl.app.mediapipe

import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.util.concurrent.ConcurrentHashMap

data class CombinedLandmarkResult(
    val timestamp: Long,
    val handResult: HandLandmarkerResult?,
    val poseResult: PoseLandmarkerResult?,
    val faceResult: FaceLandmarkerResult?,
    val preprocessingLatencyMs: Long
)

class ResultAggregator(
    private val onAggregatedResult: (CombinedLandmarkResult) -> Unit
) {

    private data class TimestampData(
        val frameStartTimeMs: Long,
        var handResult: HandLandmarkerResult? = null,
        var poseResult: PoseLandmarkerResult? = null,
        var faceResult: FaceLandmarkerResult? = null,
        var handReceived: Boolean = false,
        var poseReceived: Boolean = false,
        var faceReceived: Boolean = false
    )

    private val pendingResults = ConcurrentHashMap<Long, TimestampData>()

    fun registerTimestamp(timestamp: Long, frameStartTimeMs: Long) {
        pendingResults[timestamp] = TimestampData(frameStartTimeMs = frameStartTimeMs)
    }

    fun onHandResult(timestamp: Long, result: HandLandmarkerResult?) {
        val data = pendingResults[timestamp] ?: return
        synchronized(data) {
            data.handResult = result
            data.handReceived = true
            checkAndEmit(timestamp, data)
        }
    }

    fun onPoseResult(timestamp: Long, result: PoseLandmarkerResult?) {
        val data = pendingResults[timestamp] ?: return
        synchronized(data) {
            data.poseResult = result
            data.poseReceived = true
            checkAndEmit(timestamp, data)
        }
    }

    fun onFaceResult(timestamp: Long, result: FaceLandmarkerResult?) {
        val data = pendingResults[timestamp] ?: return
        synchronized(data) {
            data.faceResult = result
            data.faceReceived = true
            checkAndEmit(timestamp, data)
        }
    }

    private fun checkAndEmit(timestamp: Long, data: TimestampData) {
        if (data.handReceived && data.poseReceived && data.faceReceived) {
            pendingResults.remove(timestamp)

            val latency = SystemClock.uptimeMillis() - data.frameStartTimeMs
            Log.d("ResultAggregator", "Preprocessing Latency for ts $timestamp: $latency ms")

            val combined = CombinedLandmarkResult(
                timestamp = timestamp,
                handResult = data.handResult,
                poseResult = data.poseResult,
                faceResult = data.faceResult,
                preprocessingLatencyMs = latency
            )
            onAggregatedResult(combined)
        }
    }
}
