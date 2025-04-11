import React, { useState, useEffect, Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { Sky } from '../models/sky'
import { useWeb3 } from '../hooks/useWeb3'
import { useBonds } from '../hooks/useBonds'
import { ethers } from 'ethers'
// New import for bond token ABI
import BondTokenABI from '../contracts/abis/BondToken.json'

const PYBMarketPlace = () => {
  const [netWorth, setNetWorth] = useState(0)
  const [netAPY, setNetAPY] = useState(0)
  const [isRotating, setIsRotating] = useState(true)
  const [supplies, setSupplies] = useState([])
  const [assetsToSupply, setAssetsToSupply] = useState([])
  const [assetsToBorrow, setAssetsToBorrow] = useState([])
  const { account, connectWallet, contracts, signer } = useWeb3()
  const { getBondSeries, depositCollateral, withdrawCollateral, loading, error } = useBonds(contracts, signer)
  const [bondSeries, setBondSeries] = useState([])
  
  // State for deposit form
  const [depositForm, setDepositForm] = useState({
    seriesId: '',
    amount: ''
  });
  const [depositStatus, setDepositStatus] = useState({
    loading: false,
    error: null,
    success: false
  });

  // Add new state for withdraw form
  const [withdrawForm, setWithdrawForm] = useState({
    seriesId: '',
    tokenId: '',
    amount: ''
  });

  const [withdrawStatus, setWithdrawStatus] = useState({
    loading: false,
    error: null,
    success: false
  });

  // New state for NFTs
  const [myNFTs, setMyNFTs] = useState([])

  useEffect(() => {
    const fetchBondSeries = async () => {
      if (!contracts?.bondFactory) return
      
      try {
        // Fetch first 5 bond series (you can adjust this number)
        const series = []
        for (let i = 0; i < 6; i++) {
          const seriesData = await getBondSeries(i)
          if (seriesData) {
            series.push({
              id: i,
              name: `Bond Series ${i}`,
              collateralToken: seriesData.collateralToken,
              amount: seriesData.totalDeposits,
              value: seriesData.totalDeposits, // You might want to calculate this based on current price
              apy: seriesData.initialYield,
              active: seriesData.active,
              bondToken: seriesData.bondToken // Ensure bondToken is included
            })
          }
        }
        setBondSeries(series)
      } catch (err) {
        console.error('Error fetching bond series:', err)
      }
    }

    fetchBondSeries()
  }, [contracts, getBondSeries])

  // New useEffect to fetch NFTs owned by the account
  useEffect(() => {
    const fetchMyNFTs = async () => {
      if (!account || !signer || bondSeries.length === 0) return
      const nfts = []
      for (const series of bondSeries) {
        // Ensure series has bondToken property
        if (!series.bondToken) continue
        try {
          const bondTokenContract = new ethers.Contract(series.bondToken, BondTokenABI.abi, signer)
          const balance = await bondTokenContract.balanceOf(account)
          const bal = balance.toNumber()
          for (let i = 0; i < bal; i++) {
            const tokenId = await bondTokenContract.tokenOfOwnerByIndex(account, i)
            nfts.push({
              seriesId: series.id,
              seriesName: series.name,
              tokenId: tokenId.toString()
            })
          }
        } catch (err) {
          console.error('Error fetching NFTs for series', series.id, err)
        }
      }
      setMyNFTs(nfts)
    }
    fetchMyNFTs()
  }, [account, signer, bondSeries])

  // Handle deposit form changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setDepositForm(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Handle deposit submission
  const handleDeposit = async (e) => {
    e.preventDefault();
    if (!account || !signer) {
      alert('Please connect your wallet first');
      return;
    }

    setDepositStatus({ loading: true, error: null, success: false });
    try {
      // Now call depositCollateral without passing signer explicitly
      const tx = await depositCollateral(
        parseInt(depositForm.seriesId),
        depositForm.amount
      );
      await tx.wait();

      setDepositStatus({ loading: false, error: null, success: true });
      setDepositForm({ seriesId: '', amount: '' });
      const seriesData = await getBondSeries(depositForm.seriesId);
      setBondSeries(prev => prev.map(series => 
        series.id === parseInt(depositForm.seriesId)
          ? {
              ...series,
              amount: seriesData.totalDeposits,
              value: seriesData.totalDeposits
            }
          : series
      ));
    } catch (err) {
      console.error('Deposit error:', err);
      setDepositStatus({ 
        loading: false, 
        error: err.message || 'Failed to deposit collateral', 
        success: false 
      });
    }
  };

  // Add handle withdraw function
  const handleWithdraw = async (e) => {
    e.preventDefault();
    if (!account || !signer) {
      alert('Please connect your wallet first');
      return;
    }

    setWithdrawStatus({ loading: true, error: null, success: false });
    try {
      await withdrawCollateral(
        parseInt(withdrawForm.seriesId),
        parseInt(withdrawForm.tokenId),
        withdrawForm.amount
      );
      
      setWithdrawStatus({ loading: false, error: null, success: true });
      setWithdrawForm({ seriesId: '', tokenId: '', amount: '' }); // Reset form
      
      // Refresh bond series data
      const seriesData = await getBondSeries(withdrawForm.seriesId);
      setBondSeries(prev => prev.map(series => 
        series.id === parseInt(withdrawForm.seriesId) 
          ? {
              ...series,
              amount: seriesData.totalDeposits,
              value: seriesData.totalDeposits
            }
          : series
      ));
    } catch (err) {
      console.error('Withdraw error:', err);
      setWithdrawStatus({ 
        loading: false, 
        error: err.message || 'Failed to withdraw collateral', 
        success: false 
      });
    }
  };

  return (
    <div className="relative w-full min-h-screen bg-gray-900 text-white">
      <div className="absolute inset-0 z-0">
        <Canvas camera={{ position: [0, 0, 10], fov: 75 }}>
          <Suspense fallback={null}>
            <Sky isRotating={true} />
          </Suspense>
        </Canvas>
      </div>
      <div className="relative z-10 container mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">PYB Market Place</h1>
        </div>
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-2xl font-bold mb-6">Bond Series Available</h2>
          
          {loading ? (
            <div className="text-center py-4">Loading bond series...</div>
          ) : error ? (
            <div className="text-red-500 py-4">{error}</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-gray-400">
                    <th className="pb-4">Series ID</th>
                    <th className="pb-4">Collateral Token</th>
                    <th className="pb-4">Total Deposits</th>
                    <th className="pb-4">Value</th>
                    <th className="pb-4">APY</th>
                    <th className="pb-4">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {bondSeries.map((series) => (
                    <tr key={series.id} className="border-t border-gray-700">
                      <td className="py-4">{series.name}</td>
                      <td className="py-4">{series.collateralToken}</td>
                      <td className="py-4">{series.amount}</td>
                      <td className="py-4">${series.value.toLocaleString()}</td>
                      <td className="py-4">{series.apy}%</td>
                      <td className="py-4">
                        <span className={`px-2 py-1 rounded ${series.active ? 'bg-green-500' : 'bg-red-500'}`}>
                          {series.active ? 'Active' : 'Inactive'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Deposit Form */}
          <div className="mt-8 p-6 bg-gray-700 rounded-lg">
            <h3 className="text-xl font-bold mb-4">Deposit Collateral</h3>
            {!account ? (
              <button
                onClick={connectWallet}
                className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded"
              >
                Connect Wallet to Deposit
              </button>
            ) : (
              <form onSubmit={handleDeposit} className="space-y-4">
                <div>
                  <label className="block text-gray-300 mb-2">Series ID</label>
                  <input
                    type="number"
                    name="seriesId"
                    value={depositForm.seriesId}
                    onChange={handleInputChange}
                    placeholder="Enter Series ID"
                    className="w-full bg-gray-800 text-white px-4 py-2 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-gray-300 mb-2">Amount</label>
                  <input
                    type="number"
                    name="amount"
                    value={depositForm.amount}
                    onChange={handleInputChange}
                    placeholder="Enter amount to deposit"
                    step="0.000000000000000001"
                    className="w-full bg-gray-800 text-white px-4 py-2 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
                <button
                  type="submit"
                  disabled={depositStatus.loading}
                  className={`w-full ${
                    depositStatus.loading
                      ? 'bg-gray-500'
                      : 'bg-green-500 hover:bg-green-600'
                  } text-white px-4 py-2 rounded`}
                >
                  {depositStatus.loading ? 'Depositing...' : 'Deposit'}
                </button>
                
                {depositStatus.error && (
                  <div className="text-red-500 mt-2">{depositStatus.error}</div>
                )}
                {depositStatus.success && (
                  <div className="text-green-500 mt-2">
                    Deposit successful! Your transaction has been processed.
                  </div>
                )}
              </form>
            )}
          </div>

          {/* Withdraw Form */}
          <div className="mt-8 p-6 bg-gray-700 rounded-lg">
            <h3 className="text-xl font-bold mb-4">Withdraw Collateral</h3>
            {!account ? (
              <button
                onClick={connectWallet}
                className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded"
              >
                Connect Wallet to Withdraw
              </button>
            ) : (
              <form onSubmit={handleWithdraw} className="space-y-4">
                <div>
                  <label className="block text-gray-300 mb-2">Series ID</label>
                  <input
                    type="number"
                    name="seriesId"
                    value={withdrawForm.seriesId}
                    onChange={(e) => setWithdrawForm(prev => ({ ...prev, seriesId: e.target.value }))}
                    placeholder="Enter Series ID"
                    className="w-full bg-gray-800 text-white px-4 py-2 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-gray-300 mb-2">Token ID</label>
                  <input
                    type="number"
                    name="tokenId"
                    value={withdrawForm.tokenId}
                    onChange={(e) => setWithdrawForm(prev => ({ ...prev, tokenId: e.target.value }))}
                    placeholder="Enter Token ID"
                    className="w-full bg-gray-800 text-white px-4 py-2 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-gray-300 mb-2">Amount</label>
                  <input
                    type="number"
                    name="amount"
                    value={withdrawForm.amount}
                    onChange={(e) => setWithdrawForm(prev => ({ ...prev, amount: e.target.value }))}
                    placeholder="Enter amount to withdraw"
                    step="0.000000000000000001"
                    className="w-full bg-gray-800 text-white px-4 py-2 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
                <button
                  type="submit"
                  disabled={withdrawStatus.loading}
                  className={`w-full ${
                    withdrawStatus.loading
                      ? 'bg-gray-500'
                      : 'bg-red-500 hover:bg-red-600'
                  } text-white px-4 py-2 rounded`}
                >
                  {withdrawStatus.loading ? 'Withdrawing...' : 'Withdraw'}
                </button>
                
                {withdrawStatus.error && (
                  <div className="text-red-500 mt-2">{withdrawStatus.error}</div>
                )}
                {withdrawStatus.success && (
                  <div className="text-green-500 mt-2">
                    Withdrawal successful! Your transaction has been processed.
                  </div>
                )}
              </form>
            )}
          </div>

          {/* New My PYBs Section */}
          <div className="mt-8 p-6 bg-gray-700 rounded-lg">
            <h3 className="text-xl font-bold mb-4">My PYBs</h3>
            {account ? (
              myNFTs.length > 0 ? (
                <table className="w-full">
                  <thead>
                    <tr className="text-left text-gray-400">
                      <th className="pb-4">Series</th>
                      <th className="pb-4">Token ID</th>
                    </tr>
                  </thead>
                  <tbody>
                    {myNFTs.map((nft, idx) => (
                      <tr key={idx} className="border-t border-gray-700">
                        <td className="py-4">{nft.seriesName}</td>
                        <td className="py-4">{nft.tokenId}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="py-4">No PYB NFTs found.</div>
              )
            ) : (
              <div className="py-4">Connect wallet to view your PYBs.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default PYBMarketPlace