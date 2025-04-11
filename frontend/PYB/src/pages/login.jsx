import {useState, Suspense, useEffect} from 'react' 
import {Canvas} from '@react-three/fiber' 
import {useNavigate} from 'react-router-dom' 
import Loader from '../components/Loader' 
import {Sky} from '../models/sky' 
import FrontBird from '../models/FrontBird' 
import { LoginForm } from '../components/LoginComponents'  

const Login = () => {
    const navigate = useNavigate()
    const adjustBirdForScreenSize = () => {
        let screenScale, screenPosition
        
        if (window.innerWidth < 768) {
            screenScale = [0.01, 0.01, 0.01]
            screenPosition = [1, -1, 2]
        } else {
            screenScale = [0.02, 0.02, 0.02]
            screenPosition = [-3, -3.7, -5]
        }
        
        return [screenScale, screenPosition]
    }
    
    const [birdScale, birdPosition] = adjustBirdForScreenSize()
    const handleWalletConnect = (address) => {
        console.log("Wallet connected:", address)
    }
    
    return (
        <section className="w-full h-screen relative">
            <Canvas
                className="w-full h-screen bg-transparent"
                camera={{
                    position: [0, 0, 5],
                    fov: 45,
                    near: 0.1,
                    far: 1000
                }}
            >
                <Suspense fallback={<Loader/>}>
                    <directionalLight position={[1, 1, 1]} intensity={2} />
                    <ambientLight intensity={0.5}/>
                    <hemisphereLight skyColor="#b1e1ff" groundColor="#000000" intensity={1}/>
                    <Sky/>
                    <FrontBird
                        position={birdPosition}
                        scale={birdScale}
                        rotation={[0, (-Math.PI / 2)*0.75, 0]} 
                    />
                </Suspense>
            </Canvas>
            <div className="absolute top-0 right-0 h-full flex items-center pr-32">
              <LoginForm onConnect={handleWalletConnect} />
            </div>

        </section>
    )
}

export default Login