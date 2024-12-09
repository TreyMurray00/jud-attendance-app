import Image from "next/image";
import CameraComponent from "@/components/camera";
import FaceDetectionComponent from "@/components/face-detection-component";
export default function Home() {
  return (
    <div className="items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <CameraComponent></CameraComponent>
      {/* <FaceDetectionComponent></FaceDetectionComponent> */}
    </div>
  );
}
