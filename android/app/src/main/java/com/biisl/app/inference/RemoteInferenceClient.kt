package com.biisl.app.inference

import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

data class ARSubtitleResult(
    val englishSentence: String,
    val confidence: Float,
    val isStable: Boolean,
    val latencyMs: Long,
    val isSigning: Boolean = false
)

class RemoteInferenceClient(
    var serverBaseUrl: String = "http://10.0.2.2:8000" // 10.0.2.2 for Android Emulator, or local Wi-Fi IP (e.g., http://192.168.1.5:8000)
) {
    suspend fun checkHealth(): Boolean = withContext(Dispatchers.IO) {
        try {
            val url = URL("${cleanUrl(serverBaseUrl)}/health")
            val conn = url.openConnection() as HttpURLConnection
            conn.connectTimeout = 3000
            conn.readTimeout = 3000
            conn.requestMethod = "GET"
            val responseCode = conn.responseCode
            conn.disconnect()
            responseCode == 200
        } catch (e: Exception) {
            Log.e("RemoteInference", "Health check failed: ${e.message}")
            false
        }
    }

    suspend fun predictLandmarks(landmarksFlat: FloatArray): ARSubtitleResult = withContext(Dispatchers.IO) {
        val startT = SystemClock.uptimeMillis()
        try {
            val url = URL("${cleanUrl(serverBaseUrl)}/api/predict")
            val conn = url.openConnection() as HttpURLConnection
            conn.connectTimeout = 4000
            conn.readTimeout = 4000
            conn.requestMethod = "POST"
            conn.setRequestProperty("Content-Type", "application/json")
            conn.doOutput = true

            val jsonArray = JSONArray()
            landmarksFlat.forEach { jsonArray.put(it.toDouble()) }

            val payload = JSONObject().apply {
                put("landmarks", jsonArray)
                put("signer_id", "android_user")
                put("stabilize", true)
            }

            OutputStreamWriter(conn.outputStream).use { it.write(payload.toString()) }

            if (conn.responseCode == 200) {
                val reader = BufferedReader(InputStreamReader(conn.inputStream))
                val responseStr = reader.readText()
                reader.close()
                conn.disconnect()

                val resJson = JSONObject(responseStr)
                val sentence = resJson.optString("english_sentence", "Translating...")
                val conf = resJson.optDouble("confidence", 0.9).toFloat()
                val isStable = resJson.optBoolean("is_stable", true)
                val latency = SystemClock.uptimeMillis() - startT

                ARSubtitleResult(
                    englishSentence = sentence,
                    confidence = conf,
                    isStable = isStable,
                    latencyMs = latency
                )
            } else {
                conn.disconnect()
                ARSubtitleResult("Server error (${conn.responseCode})", 0.0f, false, SystemClock.uptimeMillis() - startT)
            }
        } catch (e: Exception) {
            Log.w("RemoteInference", "Predict request failed: ${e.message}")
            ARSubtitleResult(
                englishSentence = "Ready to translate. (Connecting to Wi-Fi server...)",
                confidence = 0.0f,
                isStable = false,
                latencyMs = SystemClock.uptimeMillis() - startT
            )
        }
    }

    private fun cleanUrl(raw: String): String {
        return raw.trim().removeSuffix("/")
    }
}
