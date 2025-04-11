import React, { useState, Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Sky } from "../models/sky";
import { Fox } from "../models/Fox";

const About = () => {
  const [isRotating, setIsRotating] = useState(false);
  const [currentAnimation, setCurrentAnimation] = useState("idle");

  return (
    <section className="relative w-full h-screen overflow-hidden">
      {/* 3D Background - Sky */}
      <div className="absolute inset-0 w-full h-full">
        <Canvas className="w-full h-full" camera={{ position: [0, 0, 10], fov: 75 }}>
          <directionalLight position={[1, 1, 1]} intensity={2} />
          <ambientLight intensity={0.5} />
          <Suspense fallback={null}>
            <Sky isRotating={isRotating} />
          </Suspense>
        </Canvas>
      </div>

      {/* Content Container - Fixed position, non-scrollable */}
      <div className="absolute inset-0 flex flex-col items-center p-8">
        {/* Header Section */}
        <div className="max-w-7xl mx-auto text-center mb-12 mt-8 px-4">
          <h1 className="text-5xl font-bold orange-gradient_text mb-2 drop-shadow-lg">About PyB</h1>
          <p className="text-xl text-[#8B4513] leading-relaxed drop-shadow-md">
            Perpetual Yield Bonds (PYB) is an innovative DeFi protocol that bridges traditional finance with blockchain technology. Our protocol allows users to lock their crypto assets (such as ETH, stETH, or DAI) and receive NFT-based bonds in return, complete with continuous yield streaming capabilities. These bonds are fully tradable on secondary markets, solving the liquidity problem traditional staking faces. Through our advanced StrategyManager, deposits are automatically optimized across various yield-generating protocols, while our YieldDistribution system ensures transparent and efficient streaming of yields to bondholders. The protocol features automated rebalancing via Chainlink, comprehensive security measures, and an emergency reserve system managed by our ProtocolTreasury, making PYB a secure and efficient solution for generating sustainable yields in DeFi.
          </p>
        </div>

        {/* Interactive Demo - Fox Model */}
        <div className="w-full max-w-3xl mt-auto">
          <p className="text-[#A0522D] text-md mb-4 drop-shadow-md">Click on the fox to see it move!</p>
          
          <div className="h-64 w-full">
            <Canvas camera={{ position: [0, 2, 5], fov: 45 }}>
              <directionalLight position={[0, 5, 5]} intensity={1.5} />
              <ambientLight intensity={0.5} />
              <Suspense fallback={null}>
                <Fox 
                  currentAnimation={currentAnimation} 
                  position={[0, -1, 0]} 
                  rotation={[0, Math.PI / 2, 0]} 
                  scale={[0.6, 0.6, 0.6]} 
                  onClick={() => {
                    const nextAnimation = currentAnimation === "idle" ? "walk" : 
                                          currentAnimation === "walk" ? "run" : "idle";
                    setCurrentAnimation(nextAnimation);
                  }}
                />
              </Suspense>
              <OrbitControls 
                enableZoom={false}
                enablePan={false}
                rotateSpeed={0.5}
                onChange={() => setIsRotating(true)}
                onEnd={() => setIsRotating(false)}
              />
            </Canvas>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;