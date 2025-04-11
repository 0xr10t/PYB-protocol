import {useState,Suspense} from 'react'
import {Canvas} from '@react-three/fiber'
import Loader from '../components/Loader'
import Island from '../models/island'
import {Sky} from '../models/sky'
import {Bird} from '../models/Bird'
import {Plane} from '../models/Plane'
const Home = () => {
    const [isRotating,setIsRotating]=useState(false)
    const adjustIslandForScreenSize =()=>{
        let screenScale=null
        let screenPosition=[0,-6.5,-43]
        let rotation =[0.1,4,7,0]
        if(window.innerWidth<768){
            screenScale=[0.9,0.9,0.9]
        }
        else{
            screenScale =[1,1,1]
        }
        return [screenScale,screenPosition,rotation]
        }
        const adjustplaneForScreenSize = () => {
            let screenScale, screenPosition
        
            if (window.innerWidth < 768) {
              screenScale = [1.5, 1.5, 1.5]
              screenPosition = [0, -1.5, 0]
            } else {
              screenScale = [3, 3, 3]
              screenPosition = [0, -4, -4]
            }
        
            return [screenScale, screenPosition]
          }
        const[ islandScale ,islandPosition,islandRotation]=adjustIslandForScreenSize()
        const [planeScale, planePosition] = adjustplaneForScreenSize()
  return (
         <section className='w-full h-screen relative'>
        <Canvas className="w-full h-screen bg-transparent" camera={{near:0.1 ,far :1000}}>
            <Suspense fallback={<Loader/>}>
            <directionalLight position ={[1,1,1]} intensity ={2} />
            <ambientLight intensity ={0.5}/>
            <hemisphereLight skyColor="#b1e1ff" groundcolor ="#00000" intensity={1}/>
            <Bird/>
            <Sky isRotating={isRotating} />
            <Island
            isRotating={isRotating}
            setIsRotating={setIsRotating}
            position={islandPosition}
            rotation={[0.1, 4.7077, 0]}
            scale={islandScale}
          />
    
            <Plane 
             isRotating={isRotating}
             position={planePosition}
             scale={planeScale}
             rotation={[0, 20.1, 0]}
             />
            </Suspense>
        </Canvas>
    </section>
  )
}

export default Home