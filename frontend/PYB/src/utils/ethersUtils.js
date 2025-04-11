import { ethers } from 'ethers'
import { CONTRACT_ADDRESSES } from '../contracts/addresses'
import BondTokenABI from '../contracts/BondToken.json'
import BondFactoryABI from '../contracts/BondToken.json'
import ProtocolTreasuryABI from '../contracts/BondToken.json'
import StrategyManagerABI from '../contracts/BondToken.json'
import YieldDistributionABI from '../contracts/BondToken.json'

export const getProvider = () => {
  if (!window.ethereum) throw new Error('Install MetaMask')
  return new ethers.BrowserProvider(window.ethereum)
}

export const getSigner = async () => {
  const provider = getProvider()
  return await provider.getSigner()
}

export const getContract = async (name) => {
  const signer = await getSigner()
  let abi, address

  switch (name) {
    case 'BondToken':
      abi = BondTokenABI.abi
      address = CONTRACT_ADDRESSES.BondToken
      break
    case 'BondFactory':
      abi = BondFactoryABI.abi
      address = CONTRACT_ADDRESSES.BondFactory
      break
    case 'ProtocolTreasury':
      abi = ProtocolTreasuryABI.abi
      address = CONTRACT_ADDRESSES.ProtocolTreasury
        break
    case 'StrategyManager':
       abi = StrategyManagerABI.abi
       address = CONTRACT_ADDRESSES.StrategyManager
       break
    case 'YieldDistribution':
        abi = YieldDistributionABI.abi
        address = CONTRACT_ADDRESSES.YieldDistribution
        break
    default:
      throw new Error('Unknown contract name')
  }

  return new ethers.Contract(address, abi, signer)
}
