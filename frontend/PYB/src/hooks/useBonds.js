import { useState, useCallback } from 'react';
import { ethers } from 'ethers';
import { formatAmount, parseAmount } from '../utils/contracts';
import { CONTRACT_ADDRESSES } from '../contracts/addresses';
import BondFactoryABI from '../contracts/abis/BondFactory.json';
import BondTokenABI from '../contracts/abis/BondToken.json';
import StrategyManagerABI from '../contracts/abis/StrategyManager.json';

// Add ERC20 ABI for token approval
const ERC20_ABI = [
    "function approve(address spender, uint256 amount) external returns (bool)",
    "function allowance(address owner, address spender) external view returns (uint256)",
    "function balanceOf(address account) external view returns (uint256)",
    "function transfer(address to, uint256 amount) external returns (bool)"
];

export const useBonds = (contracts, signer) => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Function to approve token spending
    const approveToken = useCallback(async (tokenAddress, amount) => {
        if (!signer) {
            throw new Error('Signer not initialized');
        }

        const tokenContract = new ethers.Contract(tokenAddress, ERC20_ABI, signer);
        
        try {
            const tx = await tokenContract.approve(
                CONTRACT_ADDRESSES.BondFactory,
                amount,
                {
                    gasLimit: 100000
                }
            );
            await tx.wait();
        } catch (err) {
            console.error('Error approving token:', err);
            throw new Error('Failed to approve token transfer');
        }
    }, [signer]);

    // Create a new bond series
    const createBondSeries = useCallback(async (collateralToken, initialYield) => {
        if (!contracts?.bondFactory || !signer) {
            setError('Contracts or signer not initialized');
            return;
        }

        setLoading(true);
        try {
            const tx = await contracts.bondFactory.createBondSeries(
                collateralToken,
                parseAmount(initialYield.toString())
            );
            const receipt = await tx.wait();
            return receipt;
        } catch (err) {
            setError(err.message);
            console.error('Error creating bond series:', err);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [contracts, signer]);

    // Deposit collateral into a bond series
    const depositCollateral = useCallback(async (seriesId, amount) => {
        if (!signer) {
            throw new Error('Signer not initialized');
        }
        setLoading(true);
        try {
            const bondFactory = new ethers.Contract(
                CONTRACT_ADDRESSES.BondFactory,
                BondFactoryABI.abi,
                signer
            );
            const series = await bondFactory.getBondSeries(seriesId);
            const collateralToken = series.collateralToken;
            const amountInWei = parseAmount(amount);
            await approveToken(collateralToken, amountInWei);
            const tx = await bondFactory.depositCollateral(
                seriesId,
                amountInWei,
                {
                    gasLimit: 300000
                }
            );
            const receipt = await tx.wait();
            return receipt;
        } catch (err) {
            console.error('Error depositing collateral:', err);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [approveToken, signer]);

    // Get bond series details
    const getBondSeries = useCallback(async (seriesId) => {
        if (!contracts?.bondFactory) {
            setError('Contracts not initialized');
            return;
        }

        setLoading(true);
        try {
            const series = await contracts.bondFactory.getBondSeries(seriesId);
            return {
                bondToken: series[0],
                collateralToken: series[1],
                initialYield: formatAmount(series[2]),
                totalDeposits: formatAmount(series[3]),
                active: series[4]
            };
        } catch (err) {
            setError(err.message);
            console.error('Error getting bond series:', err);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [contracts]);

    // Get bond details
    const getBondDetails = useCallback(async (bondTokenAddress, tokenId) => {
        if (!contracts?.bondFactory) {
            setError('Contracts not initialized');
            return;
        }

        setLoading(true);
        try {
            const bondToken = new ethers.Contract(
                bondTokenAddress,
                ['function getBondDetails(uint256) view returns (uint256,uint256,uint256,uint256)'],
                contracts.bondFactory.provider
            );
            
            const details = await bondToken.getBondDetails(tokenId);
            return {
                amount: formatAmount(details[0]),
                yieldRate: formatAmount(details[1]),
                lastClaimTimestamp: new Date(details[2].toNumber() * 1000),
                unclaimedYield: formatAmount(details[3])
            };
        } catch (err) {
            setError(err.message);
            console.error('Error getting bond details:', err);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [contracts]);

    // Claim yield from a bond
    const claimYield = useCallback(async (bondTokenAddress, tokenId) => {
        if (!contracts?.yieldDistribution || !signer) {
            setError('Contracts or signer not initialized');
            return;
        }

        setLoading(true);
        try {
            const tx = await contracts.yieldDistribution.claimYield(bondTokenAddress, tokenId);
            const receipt = await tx.wait();
            return receipt;
        } catch (err) {
            setError(err.message);
            console.error('Error claiming yield:', err);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [contracts, signer]);

    // Reinvest yield into a bond
    const reinvestYield = useCallback(async (bondTokenAddress, tokenId) => {
        if (!contracts?.yieldDistribution || !signer) {
            setError('Contracts or signer not initialized');
            return;
        }

        setLoading(true);
        try {
            const tx = await contracts.yieldDistribution.reinvestYield(bondTokenAddress, tokenId);
            const receipt = await tx.wait();
            return receipt;
        } catch (err) {
            setError(err.message);
            console.error('Error reinvesting yield:', err);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [contracts, signer]);

    // Get yield stream details
    const getYieldStream = useCallback(async (bondTokenAddress, tokenId) => {
        if (!contracts?.yieldDistribution) {
            setError('Contracts not initialized');
            return;
        }

        setLoading(true);
        try {
            const stream = await contracts.yieldDistribution.getYieldStream(bondTokenAddress, tokenId);
            return {
                totalYield: formatAmount(stream[0]),
                lastDistributionTimestamp: new Date(stream[1].toNumber() * 1000),
                yieldPerSecond: formatAmount(stream[2])
            };
        } catch (err) {
            setError(err.message);
            console.error('Error getting yield stream:', err);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [contracts]);

    // Add withdraw collateral function
    const withdrawCollateral = useCallback(async (seriesId, tokenId, amount) => {
        if (!signer) {
            throw new Error('Signer not initialized');
        }

        setLoading(true);
        try {
            // Get bond series details
            const bondFactory = new ethers.Contract(
                CONTRACT_ADDRESSES.BondFactory,
                BondFactoryABI.abi,
                signer
            );

            const series = await bondFactory.getBondSeries(seriesId);
            
            // Check if the user owns the bond token
            const bondToken = new ethers.Contract(
                series.bondToken,
                BondTokenABI.abi,
                signer
            );
            
            const owner = await bondToken.ownerOf(tokenId);
            if (owner.toLowerCase() !== (await signer.getAddress()).toLowerCase()) {
                throw new Error('You do not own this bond token');
            }

            // Get bond details to check available amount
            const bondDetails = await bondToken.getBondDetails(tokenId);
            if (bondDetails.amount < amount) {
                throw new Error('Insufficient bond amount');
            }

            // Create StrategyManager contract instance
            const strategyManager = new ethers.Contract(
                CONTRACT_ADDRESSES.StrategyManager,
                StrategyManagerABI.abi,
                signer
            );

            // First, withdraw from strategy
            const withdrawTx = await strategyManager.rebalanceStrategy(
                series.collateralToken,
                seriesId,
                {
                    gasLimit: 300000
                }
            );
            await withdrawTx.wait();

            // Then burn the bond token
            const burnTx = await bondToken.burn(tokenId, amount, {
                gasLimit: 200000
            });
            await burnTx.wait();

            // Finally transfer the tokens back to the user
            const tokenContract = new ethers.Contract(
                series.collateralToken,
                ERC20_ABI,
                signer
            );
            const transferTx = await tokenContract.transfer(
                await signer.getAddress(),
                amount,
                {
                    gasLimit: 100000
                }
            );
            await transferTx.wait();

            return true;
        } catch (err) {
            console.error('Error withdrawing collateral:', err);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [signer]);

    return {
        loading,
        error,
        createBondSeries,
        depositCollateral,
        withdrawCollateral,
        getBondSeries,
        getBondDetails,
        claimYield,
        reinvestYield,
        getYieldStream,
        approveToken
    };
};