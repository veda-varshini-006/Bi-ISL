package com.biisl.app.inference

import android.content.Context
import android.os.Debug
import java.io.File

data class BenchmarkMetrics(
    var loadTimeMs: Long = 0L,
    var warmUpTimeMs: Long = 0L,
    var p50LatencyMs: Long = 0L,
    var p95LatencyMs: Long = 0L,
    var memoryUsageMb: Float = 0f,
    var modelSizeMb: Float = 0f
)

class BenchmarkStats {
    private val latencies = mutableListOf<Long>()

    fun recordLatency(timeMs: Long) {
        latencies.add(timeMs)
    }

    fun calculatePercentiles(metrics: BenchmarkMetrics) {
        if (latencies.isEmpty()) return
        
        val sorted = latencies.sorted()
        
        val p50Index = (sorted.size * 0.50).toInt().coerceAtMost(sorted.size - 1)
        val p95Index = (sorted.size * 0.95).toInt().coerceAtMost(sorted.size - 1)

        metrics.p50LatencyMs = sorted[p50Index]
        metrics.p95LatencyMs = sorted[p95Index]
    }

    fun measureMemoryUsage(metrics: BenchmarkMetrics) {
        val memInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(memInfo)
        // totalPss is in KB
        metrics.memoryUsageMb = memInfo.totalPss / 1024f
    }

    fun measureModelSize(context: Context, modelFileName: String, metrics: BenchmarkMetrics) {
        try {
            // Check if it's in assets first (common for testing)
            val assetFd = context.assets.openFd(modelFileName)
            metrics.modelSizeMb = assetFd.length / (1024f * 1024f)
            assetFd.close()
        } catch (e: Exception) {
            // Fallback to checking normal file path
            val file = File(modelFileName)
            if (file.exists()) {
                metrics.modelSizeMb = file.length() / (1024f * 1024f)
            }
        }
    }
    
    fun clear() {
        latencies.clear()
    }
}
