'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { Button } from "@/components/ui/button"
import * as blazeface from '@tensorflow-models/blazeface'
import * as tf from '@tensorflow/tfjs-core'
import '@tensorflow/tfjs-backend-webgl'

interface FaceDetection {
  topLeft: [number, number];
  bottomRight: [number, number];
  probability: [number];
  landmarks: [number, number][];
}

export default function CameraComponent() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [model, setModel] = useState<blazeface.BlazeFaceModel | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [predictions, setPredictions] = useState<FaceDetection[]>([])
  const [debugInfo, setDebugInfo] = useState<string>('')

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
      setDebugInfo('')
    }
  }

  const loadModel = async () => {
    try {
      await tf.setBackend('webgl')
      const loadedModel = await blazeface.load()
      setModel(loadedModel)
      setIsLoading(false)
    } catch (err) {
      console.error('Error loading the face detection model:', err)
      setError('Failed to load the face detection model.')
      setIsLoading(false)
    }
  }

  let frameCount = 0;

  const detectFaces = useCallback(async () => {
    if (videoRef.current && model && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
    
      if (!ctx) return;

      // Ensure video is ready
      if (video.readyState !== video.HAVE_ENOUGH_DATA) {
        requestAnimationFrame(detectFaces);
        return;
      }

      // Set canvas dimensions to match the displayed video size
      canvas.width = video.clientWidth;
      canvas.height = video.clientHeight;

      // Calculate scale factors
      const scaleX = video.clientWidth / video.videoWidth;
      const scaleY = video.clientHeight / video.videoHeight;

      try {
        const returnTensors = false;
        const predictions = await model.estimateFaces(video, returnTensors) as unknown as FaceDetection[];
        setPredictions(predictions);
        frameCount++;
        setDebugInfo(`Frame: ${frameCount}, Raw predictions: ${JSON.stringify(predictions)}`);

        // Clear previous drawings
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw face detections
        ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
        ctx.strokeStyle = '#FF0000';
        ctx.lineWidth = 2;
        
        predictions.forEach((face) => {
          const [x, y] = face.topLeft;
          const [width, height] = [
            face.bottomRight[0] - face.topLeft[0],
            face.bottomRight[1] - face.topLeft[1]
          ];

          // Draw bounding box
          ctx.strokeRect(x * scaleX, y * scaleY, width * scaleX, height * scaleY);

          // Draw landmarks
          face.landmarks.forEach((landmark) => {
            ctx.beginPath();
            ctx.arc(landmark[0] * scaleX, landmark[1] * scaleY, 3, 0, 2 * Math.PI);
            ctx.fill();
          });
        });

      } catch (err) {
        console.error('Error during face detection:', err);
        setError('Face detection failed. Please try again.');
        setDebugInfo(`Error: ${err}`);
      }
    }
  }, [model, frameCount, setDebugInfo, setPredictions, setError]);

  useEffect(() => {
    loadModel()
    return () => {
      stopCamera()
    }
  }, [])

  useEffect(() => {
    let timeoutId: NodeJS.Timeout;

    const runDetection = async () => {
      await detectFaces();
      timeoutId = setTimeout(runDetection, 100); // Run detection every 1000ms (1 second)
    };

    if (isStreaming && model) {
      runDetection();
    }

    return () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [isStreaming, model]);

  useEffect(() => {
    const handleResize = () => {
      if (canvasRef.current && videoRef.current) {
        canvasRef.current.width = videoRef.current.clientWidth;
        canvasRef.current.height = videoRef.current.clientHeight;
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
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
              Loading face detection model...
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
        {isStreaming && model && (
          <div className="bg-gray-100 p-4 rounded-lg">
            <h2 className="text-lg font-semibold mb-2">Face Detection Results:</h2>
            {predictions.length > 0 ? (
              <ul className="max-h-60 overflow-y-auto">
                {predictions.map((face, index) => (
                  <li key={index} className="mb-2">
                    <strong>Face {index + 1}:</strong><br />
                    Probability: {face.probability[0].toFixed(4)}<br />
                    Top-left: ({face.topLeft[0].toFixed(2)}, {face.topLeft[1].toFixed(2)})<br />
                    Bottom-right: ({face.bottomRight[0].toFixed(2)}, {face.bottomRight[1].toFixed(2)})<br />
                    Landmarks:<br />
                    - Right eye: ({face.landmarks[0][0].toFixed(2)}, {face.landmarks[0][1].toFixed(2)})<br />
                    - Left eye: ({face.landmarks[1][0].toFixed(2)}, {face.landmarks[1][1].toFixed(2)})<br />
                    - Nose: ({face.landmarks[2][0].toFixed(2)}, {face.landmarks[2][1].toFixed(2)})<br />
                    - Mouth: ({face.landmarks[3][0].toFixed(2)}, {face.landmarks[3][1].toFixed(2)})<br />
                    - Right ear: ({face.landmarks[4][0].toFixed(2)}, {face.landmarks[4][1].toFixed(2)})<br />
                    - Left ear: ({face.landmarks[5][0].toFixed(2)}, {face.landmarks[5][1].toFixed(2)})
                  </li>
                ))}
              </ul>
            ) : (
              <p>No faces detected</p>
            )}
          </div>
        )}
        <div className="bg-gray-100 p-4 rounded-lg">
          <h3 className="text-md font-semibold mb-2">Debug Information:</h3>
          <pre className="whitespace-pre-wrap text-xs max-h-40 overflow-y-auto">{debugInfo}</pre>
        </div>
      </div>
    </div>
  )
}

