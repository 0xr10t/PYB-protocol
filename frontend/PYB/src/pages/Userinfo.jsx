import React, { useState, useEffect, Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import {Sky} from '../models/sky'

const PYBMarketPlace = () => {
  const [netWorth, setNetWorth] = useState(0)
  const [netAPY, setNetAPY] = useState(0)
  const [isRotating, setIsRotating] = useState(true)
  const [supplies, setSupplies] = useState([])
  const [assetsToSupply, setAssetsToSupply] = useState([])
  const [assetsToBorrow, setAssetsToBorrow] = useState([])
  useEffect(() => {
    setSupplies([
      { name: 'Ethereum', amount: 2.5, value: 5000, apy: 3.2 },
      { name: 'USDC', amount: 1000, value: 1000, apy: 4.5 }
    ])

    setAssetsToSupply([
      { name: 'Bitcoin', available: 0.5, apy: 2.8 },
      { name: 'Solana', available: 100, apy: 5.1 },
      { name: 'USDT', available: 5000, apy: 4.2 }
    ])

    setAssetsToBorrow([
      { name: 'Ethereum', available: 10, apy: 7.5 },
      { name: 'USDC', available: 10000, apy: 6.2 },
      { name: 'AVAX', available: 500, apy: 8.0 }
    ])

    setNetWorth(6000)
    setNetAPY(3.8)
  }, [])
  const handleSupply = (asset) => {
    console.log(`Supplying ${asset.name}`)
  }
  const handleBorrow = (asset) => {
    console.log(`Borrowing ${asset.name}`)
  }

  return (
    <div className="relative w-full min-h-screen bg-gray-900 text-white">
      <div className="absolute inset-0 z-0">
        <Canvas camera={{ position: [0, 0, 10], fov: 75 }}>
          <Suspense fallback={null}>
            <Sky isRotating={isRotating} />
          </Suspense>
        </Canvas>
      </div>
      <div className="relative z-10 container mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">PYB Market Place</h1>
          <div className="flex items-center gap-8">
            <div className="bg-gray-800 bg-opacity-70 p-4 rounded-lg">
              <p className="text-sm text-gray-400">Net Worth</p>
              <p className="text-xl font-semibold">${netWorth.toLocaleString()}</p>
            </div>
            <div className="bg-gray-800 bg-opacity-70 p-4 rounded-lg">
              <p className="text-sm text-gray-400">Net APY</p>
              <p className="text-xl font-semibold">{netAPY}%</p>
            </div>
          </div>
        </div>
        <div className="bg-gray-800 bg-opacity-70 rounded-lg p-6 mb-6">
          <h2 className="text-xl font-bold mb-4">Your Supplies</h2>
          {supplies.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-gray-400">
                    <th className="pb-4">Asset</th>
                    <th className="pb-4">Amount</th>
                    <th className="pb-4">Value</th>
                    <th className="pb-4">APY</th>
                  </tr>
                </thead>
                <tbody>
                  {supplies.map((asset, index) => (
                    <tr key={index} className="border-t border-gray-700">
                      <td className="py-4">{asset.name}</td>
                      <td className="py-4">{asset.amount}</td>
                      <td className="py-4">${asset.value.toLocaleString()}</td>
                      <td className="py-4">{asset.apy}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-gray-400">No supplies yet</p>
          )}
        </div>

        <div className="bg-gray-800 bg-opacity-70 rounded-lg p-6 mb-6">
          <h2 className="text-xl font-bold mb-4">Assets to Supply</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400">
                  <th className="pb-4">Asset</th>
                  <th className="pb-4">Available</th>
                  <th className="pb-4">APY</th>
                  <th className="pb-4">Action</th>
                </tr>
              </thead>
              <tbody>
                {assetsToSupply.map((asset, index) => (
                  <tr key={index} className="border-t border-gray-700">
                    <td className="py-4">{asset.name}</td>
                    <td className="py-4">{asset.available}</td>
                    <td className="py-4">{asset.apy}%</td>
                    <td className="py-4">
                      <button 
                        onClick={() => handleSupply(asset)}
                        className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded"
                      >
                        Supply
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="bg-gray-800 bg-opacity-70 rounded-lg p-6">
          <h2 className="text-xl font-bold mb-4">Assets to Borrow</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400">
                  <th className="pb-4">Asset</th>
                  <th className="pb-4">Available</th>
                  <th className="pb-4">APY</th>
                  <th className="pb-4">Action</th>
                </tr>
              </thead>
              <tbody>
                {assetsToBorrow.map((asset, index) => (
                  <tr key={index} className="border-t border-gray-700">
                    <td className="py-4">{asset.name}</td>
                    <td className="py-4">{asset.available}</td>
                    <td className="py-4">{asset.apy}%</td>
                    <td className="py-4">
                      <button 
                        onClick={() => handleBorrow(asset)}
                        className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded"
                      >
                        Borrow
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PYBMarketPlace