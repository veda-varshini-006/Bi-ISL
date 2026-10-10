package com.biisl.app.inference

/**
 * Interface separating presentation from Continuous ISL ML inference logic.
 */
interface InferenceEngine {
    fun initialize(modelPath: String)
    fun processLandmarkSequence(landmarksFlat: FloatArray): String?
    fun release()
}
