import { ethers } from 'ethers';
import { CONTRACT_ADDRESSES } from '../contracts/addresses';

// Import ABIs
import BondTokenABI from '../contracts/abis/BondToken.json';
import BondFactoryABI from '../contracts/abis/BondFactory.json';
import ProtocolTreasuryABI from '../contracts/abis/ProtocolTreasury.json';
import StrategyManagerABI from '../contracts/abis/StrategyManager.json';
import YieldDistributionABI from '../contracts/abis/YieldDistribution.json';

// Initialize contract instances
export const initializeContracts = (provider) => {
    if (!provider) {
        console.error('Provider is required to initialize contracts');
        return null;
    }

    try {
        const bondFactory = new ethers.Contract(
            CONTRACT_ADDRESSES.BondFactory,
            BondFactoryABI.abi, // Access the ABI from the JSON file
            provider
        );

        const protocolTreasury = new ethers.Contract(
            CONTRACT_ADDRESSES.ProtocolTreasury,
            ProtocolTreasuryABI.abi,
            provider
        );

        const strategyManager = new ethers.Contract(
            CONTRACT_ADDRESSES.StrategyManager,
            StrategyManagerABI.abi,
            provider
        );

        const yieldDistribution = new ethers.Contract(
            CONTRACT_ADDRESSES.YieldDistribution,
            YieldDistributionABI.abi,
            provider
        );

        return {
            bondFactory,
            protocolTreasury,
            strategyManager,
            yieldDistribution
        };
    } catch (error) {
        console.error('Error initializing contracts:', error);
        return null;
    }
};

// Bond Factory Functions
export const createBondSeries = async (signer, collateralToken, initialYield) => {
    const bondFactory = new ethers.Contract(
        CONTRACT_ADDRESSES.BondFactory,
        BondFactoryABI.abi,
        signer
    );
    const tx = await bondFactory.createBondSeries(collateralToken, initialYield);
    return await tx.wait();
};

export const depositCollateral = async (signer, seriesId, amount) => {
    const bondFactory = new ethers.Contract(
        CONTRACT_ADDRESSES.BondFactory,
        BondFactoryABI.abi,
        signer
    );
    const tx = await bondFactory.depositCollateral(seriesId, amount);
    return await tx.wait();
};

export const getBondSeries = async (provider, seriesId) => {
    const bondFactory = new ethers.Contract(
        CONTRACT_ADDRESSES.BondFactory,
        BondFactoryABI.abi,
        provider
    );
    return await bondFactory.getBondSeries(seriesId);
};

// Bond Token Functions
export const getBondDetails = async (provider, bondTokenAddress, tokenId) => {
    const bondToken = new ethers.Contract(
        bondTokenAddress,
        BondTokenABI.abi,
        provider
    );
    return await bondToken.getBondDetails(tokenId);
};

export const claimBondYield = async (signer, bondTokenAddress, tokenId) => {
    const bondToken = new ethers.Contract(
        bondTokenAddress,
        BondTokenABI.abi,
        signer
    );
    const tx = await bondToken.claimYield(tokenId);
    return await tx.wait();
};

// Yield Distribution Functions
export const claimYield = async (signer, bondTokenAddress, tokenId) => {
    const yieldDistribution = new ethers.Contract(
        CONTRACT_ADDRESSES.YieldDistribution,
        YieldDistributionABI.abi,
        signer
    );
    const tx = await yieldDistribution.claimYield(bondTokenAddress, tokenId);
    return await tx.wait();
};

export const reinvestYield = async (signer, bondTokenAddress, tokenId) => {
    const yieldDistribution = new ethers.Contract(
        CONTRACT_ADDRESSES.YieldDistribution,
        YieldDistributionABI.abi,
        signer
    );
    const tx = await yieldDistribution.reinvestYield(bondTokenAddress, tokenId);
    return await tx.wait();
};

export const getYieldStream = async (provider, bondTokenAddress, tokenId) => {
    const yieldDistribution = new ethers.Contract(
        CONTRACT_ADDRESSES.YieldDistribution,
        YieldDistributionABI.abi,
        provider
    );
    return await yieldDistribution.getYieldStream(bondTokenAddress, tokenId);
};

// Strategy Manager Functions
export const getOptimalAmount = async (provider, token, seriesId) => {
    const strategyManager = new ethers.Contract(
        CONTRACT_ADDRESSES.StrategyManager,
        StrategyManagerABI.abi,
        provider
    );
    return await strategyManager.calculateOptimalAmount(token, seriesId);
};

// Protocol Treasury Functions
export const getReserves = async (provider, token) => {
    const protocolTreasury = new ethers.Contract(
        CONTRACT_ADDRESSES.ProtocolTreasury,
        ProtocolTreasuryABI.abi,
        provider
    );
    return await protocolTreasury.getReserves(token);
};

// Helper function to format amounts from wei to ether
export const formatAmount = (amount, decimals = 18) => {
    return ethers.formatUnits(amount, decimals);
};

// Helper function to parse amounts from ether to wei
export const parseAmount = (amount, decimals = 18) => {
    return ethers.parseUnits(amount, decimals);
}; 