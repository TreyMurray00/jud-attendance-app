'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import * as faceapi from 'face-api.js'

export default function FaceDetectionComponent() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [faceDescriptors, setFaceDescriptors] = useState<number[][]>([])

  useEffect(() => {
    const loadModels = async () => {
      await faceapi.nets.tinyFaceDetector.loadFromUri('/models')
      await faceapi.nets.faceLandmark68Net.loadFromUri('/models')
      await faceapi.nets.faceRecognitionNet.loadFromUri('/models')
    }
    loadModels()
  }, [])

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsStreaming(true)
        setError(null)
      }
    } catch (err) {
      console.error('Error accessing the camera:', err)
      setError('Failed to access the camera. Please make sure you have given permission.')
    }
  }

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
      tracks.forEach(track => track.stop())
      videoRef.current.srcObject = null
      setIsStreaming(false)
      setFaceDescriptors([])
    }
  }

  useEffect(() => {
    let animationFrameId: number

    const detectFaces = async () => {
      if (videoRef.current && canvasRef.current) {
        const detections = await faceapi.detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks()
          .withFaceDescriptors()

        const displaySize = { width: videoRef.current.width, height: videoRef.current.height }
        faceapi.matchDimensions(canvasRef.current, displaySize)

        const resizedDetections = faceapi.resizeResults(detections, displaySize)

        canvasRef.current.getContext('2d')?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
        faceapi.draw.drawDetections(canvasRef.current, resizedDetections)
        faceapi.draw.drawFaceLandmarks(canvasRef.current, resizedDetections)

        setFaceDescriptors(resizedDetections.map(d => Array.from(d.descriptor)))

        animationFrameId = requestAnimationFrame(detectFaces)
      }
    }

    if (isStreaming) {
      detectFaces()
    }

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId)
      }
    }
  }, [isStreaming])

  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  return (
    <div className="flex flex-col items-center space-y-4">
      <div className="relative w-full max-w-md aspect-video bg-gray-200 rounded-lg overflow-hidden">
        {error && (
          <div className="absolute inset-0 flex items-center justify-center text-red-500 text-center p-4">
            {error}
          </div>
        )}
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          autoPlay
          playsInline
          muted
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full"
        />
      </div>
      <div className="flex space-x-4">
        <Button onClick={startCamera} disabled={isStreaming}>
          Start Camera
        </Button>
        <Button onClick={stopCamera} disabled={!isStreaming} variant="outline">
          Stop Camera
        </Button>
      </div>
      <div className="w-full max-w-md">
        <h2 className="text-lg font-semibold mb-2">Face Descriptors:</h2>
        <div className="max-h-60 overflow-y-auto bg-gray-100 p-4 rounded-lg">
          {faceDescriptors.map((descriptor, index) => (
            <div key={index} className="mb-2">
              <h3 className="font-medium">Face {index + 1}:</h3>
              <p className="text-xs break-all">{JSON.stringify(descriptor)}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

