import "@mediapipe/face_mesh"
import "@tensorflow/tfjs-core"
import "@tensorflow/tfjs-backend-webgl"
import * as faceLandMarkDetection from "@tensorflow-models/face-landmarks-detection"

const model = faceLandMarkDetection.SupportedModels.MediaPipeFaceMesh
const detectorConfig = {
    runtime: "mediapipe",
    solutionPath: "../node_modules/@mediapipe/face_mesh" 
}

const detector = await faceLandMarkDetection.createDetector(model,detectorConfig)