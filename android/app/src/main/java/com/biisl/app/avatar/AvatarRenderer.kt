package com.biisl.app.avatar

/**
 * Abstract interface separating presentation from 3D Avatar Rendering logic.
 */
interface AvatarRenderer {
    fun initialize(modelPath: String)
    fun renderISL(islRepresentation: String)
    fun stopRendering()
    fun release()
}
