package com.biisl.app.inference

/**
 * Abstract interface separating presentation from ML inference logic.
 */
interface InferenceEngine {
    fun initialize(modelPath: String)
    fun processFrame(frameData: ByteArray, width: Int, height: Int): String?
    fun translateTextToISL(text: String): String
    fun release()
}
