import { useEffect, useRef } from "react"
import { useFrame } from "@react-three/fiber"
import { useAnimations, useGLTF } from "@react-three/drei"

import birdScene from "../assets/3d/bird.glb"

export function FrontBird({ position, scale, rotation, onMouthOpen }) {
  const birdRef = useRef()
  
  const { scene, animations } = useGLTF(birdScene)
  const { actions } = useAnimations(animations, birdRef)
  const mouthOpenRef = useRef(false)
  useEffect(() => {

    console.log("Available animations:", Object.keys(actions))

    if (actions["Take 001"]) {
      actions["Take 001"].play()
    } else {
      const firstAnimation = Object.keys(actions)[0]
      if (firstAnimation) {
        console.log(`Playing animation: ${firstAnimation}`)
        actions[firstAnimation].play()
      } else {
        console.error("No animations found in the model")
      }
    }
  }, [actions])
  
  useFrame(({ clock }) => {
    if (birdRef.current) {
      const time = clock.getElapsedTime()
      birdRef.current.position.y = position[1] + Math.sin(time * 0.3) * 0.1
      birdRef.current.rotation.z = Math.sin(time * 0.2) * 0.05
      const shouldOpenMouth = Math.sin(time * 0.4) > 0.6
      
      if (shouldOpenMouth && !mouthOpenRef.current) {
        mouthOpenRef.current = true
        onMouthOpen(true)
      } else if (!shouldOpenMouth && mouthOpenRef.current) {
        mouthOpenRef.current = false
        onMouthOpen(false)
      }
    }
  })
  
  return (
    <group ref={birdRef} position={position} scale={scale} rotation={rotation}>
      <primitive object={scene} />
    </group>
  )
}

export default FrontBird