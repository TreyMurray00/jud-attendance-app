'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { Button } from "@/components/ui/button"
import * as blazeface from '@tensorflow-models/blazeface'
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection'
import * as tf from '@tensorflow/tfjs-core'
import '@tensorflow/tfjs-backend-webgl'

interface FaceDetection {
  topLeft: [number, number];
  bottomRight: [number, number];
  probability: [number];
  landmarks: [number, number][];
}

interface CroppedFace {
  image: string;
  landmarks: faceLandmarksDetection.Face[];
}

export default function CameraComponent() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [blazefaceModel, setBlazefaceModel] = useState<blazeface.BlazeFaceModel | null>(null)
  const [landmarkModel, setLandmarkModel] = useState<faceLandmarksDetection.FaceLandmarksDetector | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [predictions, setPredictions] = useState<FaceDetection[]>([])
  const [croppedFaces, setCroppedFaces] = useState<CroppedFace[]>([])
  const [debugInfo, setDebugInfo] = useState<string>('')

  // Add scaleFactor and yScalerPos variables
  const scaleFactor = 2
  const yScalerPos = 0.85

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play()
          setIsStreaming(true)
        }
        setError(null)
      }
    } catch (err) {
      console.error('Error accessing the camera:', err)
      setError('Failed to access the camera. Please make sure you have given permission.')
      setIsLoading(false)
    }
  }

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
      tracks.forEach(track => track.stop())
      videoRef.current.srcObject = null
      setIsStreaming(false)
      setPredictions([])
      setCroppedFaces([])
      setDebugInfo('')
    }
  }

  const loadModels = async () => {
    try {
      await tf.setBackend('webgl')
      const blazefaceLoadedModel = await blazeface.load()
      setBlazefaceModel(blazefaceLoadedModel)
      
      const landmarkLoadedModel = await faceLandmarksDetection.createDetector(
        faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
        {
          runtime: 'tfjs',
          refineLandmarks: true,
          maxFaces: 1
        }
      )
      setLandmarkModel(landmarkLoadedModel)
      
      setIsLoading(false)
      setDebugInfo('Models loaded successfully')
    } catch (err) {
      console.error('Error loading the models:', err)
      setError('Failed to load the face detection models.')
      setDebugInfo(`Error loading models: ${err}`)
      setIsLoading(false)
    }
  }

  const cropFace = (video: HTMLVideoElement, face: FaceDetection): HTMLCanvasElement => {
    const tempCanvas = document.createElement('canvas')
    const tempCtx = tempCanvas.getContext('2d')
    if (!tempCtx) return tempCanvas

    const [x, y] = face.topLeft
    const width = face.bottomRight[0] - face.topLeft[0]
    const height = face.bottomRight[1] - face.topLeft[1]

    // Apply scaleFactor and yScalerPos to cropping
    const scaledWidth = width * scaleFactor
    const scaledHeight = height * scaleFactor
    const scaledX = x - (scaledWidth - width) / 2
    const scaledY = y * yScalerPos - (scaledHeight - height) / 2

    tempCanvas.width = scaledWidth
    tempCanvas.height = scaledHeight
    tempCtx.drawImage(video, scaledX, scaledY, scaledWidth, scaledHeight, 0, 0, scaledWidth, scaledHeight)

    return tempCanvas
  }

  const detectFaces = useCallback(async () => {
    if (videoRef.current && blazefaceModel && canvasRef.current && landmarkModel) {
      const video = videoRef.current
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
    
      if (!ctx) return

      if (video.readyState !== video.HAVE_ENOUGH_DATA) {
        requestAnimationFrame(detectFaces)
        return
      }

      canvas.width = video.clientWidth
      canvas.height = video.clientHeight

      const scaleX = video.clientWidth / video.videoWidth
      const scaleY = video.clientHeight / video.videoHeight

      try {
        const predictions = await blazefaceModel.estimateFaces(video, false) as unknown as FaceDetection[]
        setPredictions(predictions)

        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.fillStyle = 'rgba(255, 0, 0, 0.5)'
        ctx.strokeStyle = '#FF0000'
        ctx.lineWidth = 2

        const newCroppedFaces: CroppedFace[] = []

        for (const face of predictions) {
          const [x, y] = face.topLeft
          const [width, height] = [
            face.bottomRight[0] - face.topLeft[0],
            face.bottomRight[1] - face.topLeft[1]
          ]

          // Note: scaleFactor is not applied to the bounding box, only yScalerPos to y coordinate
          ctx.strokeRect(
            x * scaleX, 
            y * scaleY * yScalerPos, 
            width * scaleX, 
            height * scaleY
          )

          face.landmarks.forEach((landmark) => {
            ctx.beginPath()
            ctx.arc(landmark[0] * scaleX, landmark[1] * scaleY, 3, 0, 2 * Math.PI)
            ctx.fill()
          })

          const croppedCanvas = cropFace(video, face)
          const landmarks = await landmarkModel.estimateFaces(croppedCanvas)
          newCroppedFaces.push({ image: croppedCanvas.toDataURL(), landmarks })
        }

        setCroppedFaces(newCroppedFaces)
        setDebugInfo(`Detected ${predictions.length} faces at ${new Date().toLocaleTimeString()}. Scaling: yScalerPos=${yScalerPos}, scaleFactor=${scaleFactor}
Predictions:
${JSON.stringify(predictions, null, 2)}`)

      } catch (err) {
        console.error('Error during face detection:', err)
        setError('Face detection failed. Please try again.')
        setDebugInfo(`Error: ${err}`)
      }
    }
  }, [blazefaceModel, landmarkModel, scaleFactor, yScalerPos])

  useEffect(() => {
    loadModels()
    return () => {
      stopCamera()
    }
  }, [])

  useEffect(() => {
    let timeoutId: NodeJS.Timeout;

    const runDetection = () => {
      detectFaces();
      timeoutId = setTimeout(runDetection, 600); // Run detection every 600ms (0.6 second)
    };

    if (isStreaming && blazefaceModel && landmarkModel) {
      runDetection();
    }

    return () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [isStreaming, blazefaceModel, landmarkModel, detectFaces]);

  return (
    <div className="flex flex-col space-y-4">
      <div className="flex flex-row space-x-4">
        <div className="w-3/5">
          <div className="relative w-full aspect-video bg-gray-200 rounded-lg overflow-hidden">
            {error && (
              <div className="absolute inset-0 flex items-center justify-center text-red-500 text-center p-4 z-20">
                {error}
              </div>
            )}
            {isLoading && (
              <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-center p-4 z-20">
                Loading face detection models...
              </div>
            )}
            <video
              ref={videoRef}
              className="absolute top-0 left-0 w-full h-full object-cover"
              autoPlay
              playsInline
              muted
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full z-10"
              style={{ pointerEvents: 'none' }}
            />
          </div>
          <div className="flex space-x-4 mt-4">
            <Button onClick={startCamera} disabled={isStreaming || isLoading}>
              Start Camera
            </Button>
            <Button onClick={stopCamera} disabled={!isStreaming || isLoading} variant="outline">
              Stop Camera
            </Button>
          </div>
        </div>
        <div className="w-2/5 space-y-4">
          <div className="bg-gray-100 p-4 rounded-lg">
            <h2 className="text-lg font-semibold mb-2">Debug Information:</h2>
            <pre className="whitespace-pre-wrap text-xs max-h-40 overflow-y-auto">{debugInfo}</pre>
            <pre className="whitespace-pre-wrap text-xs max-h-40 overflow-y-auto">{JSON.stringify(predictions)}</pre>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-4">
        {croppedFaces.map((face, index) => (
          <div key={index} className="bg-gray-100 p-4 rounded-lg">
            <h3 className="text-md font-semibold mb-2">Face {index + 1}</h3>
            <img src={face.image} alt={`Cropped face ${index + 1}`} className="w-full mb-2" />
            <div className="text-xs">
              <h4 className="font-semibold">Landmarks:</h4>
              <pre className="whitespace-pre-wrap max-h-40 overflow-y-auto">
                {JSON.stringify(face.landmarks[0], null, 2)}
              </pre>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

