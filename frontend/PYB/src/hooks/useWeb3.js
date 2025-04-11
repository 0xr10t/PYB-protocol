import { useState, useEffect } from 'react';
import { ethers } from 'ethers';
import { initializeContracts } from '../utils/contracts';
import { getNetworkConfig, RPC_URLS, DEFAULT_NETWORK } from '../config/network';

export const useWeb3 = () => {
    const [account, setAccount] = useState(null);
    const [provider, setProvider] = useState(null);
    const [signer, setSigner] = useState(null);
    const [contracts, setContracts] = useState(null);
    const [chainId, setChainId] = useState(null);
    const [isConnecting, setIsConnecting] = useState(false);
    const [error, setError] = useState(null);

    // Initialize provider
    useEffect(() => {
        const initProvider = async () => {
            try {
                // Check if MetaMask is installed
                if (window.ethereum) {
                    const provider = new ethers.BrowserProvider(window.ethereum);
                    setProvider(provider);

                    // Get network
                    const network = await provider.getNetwork();
                    setChainId(network.chainId.toString());

                    // Initialize contracts with provider
                    const contractInstances = initializeContracts(provider);
                    if (contractInstances) {
                        setContracts(contractInstances);
                    }

                    // Check if already connected
                    const accounts = await provider.listAccounts();
                    if (accounts.length > 0) {
                        setAccount(accounts[0]);
                        const signer = await provider.getSigner();
                        setSigner(signer);
                    }
                } else {
                    // If MetaMask is not installed, use a read-only provider
                    const provider = new ethers.JsonRpcProvider(RPC_URLS[DEFAULT_NETWORK]);
                    setProvider(provider);
                    
                    // Initialize contracts with read-only provider
                    const contractInstances = initializeContracts(provider);
                    if (contractInstances) {
                        setContracts(contractInstances);
                    }
                }
            } catch (err) {
                setError(err.message);
                console.error('Error initializing provider:', err);
            }
        };

        initProvider();
    }, []);

    // Handle account changes
    useEffect(() => {
        if (window.ethereum) {
            const handleAccountsChanged = (accounts) => {
                if (accounts.length > 0) {
                    setAccount(accounts[0]);
                } else {
                    setAccount(null);
                    setSigner(null);
                }
            };

            const handleChainChanged = (chainId) => {
                window.location.reload();
            };

            window.ethereum.on('accountsChanged', handleAccountsChanged);
            window.ethereum.on('chainChanged', handleChainChanged);

            return () => {
                window.ethereum.removeListener('accountsChanged', handleAccountsChanged);
                window.ethereum.removeListener('chainChanged', handleChainChanged);
            };
        }
    }, []);

    // Connect wallet
    const connectWallet = async () => {
        if (!window.ethereum) {
            setError('Please install MetaMask');
            return;
        }

        setIsConnecting(true);
        try {
            const provider = new ethers.BrowserProvider(window.ethereum);
            const accounts = await provider.send('eth_requestAccounts', []);
            setAccount(accounts[0]);
            
            const signer = await provider.getSigner();
            setSigner(signer);
            
            const contractInstances = initializeContracts(provider);
            if (contractInstances) {
                setContracts(contractInstances);
            }
        } catch (err) {
            setError(err.message);
            console.error('Error connecting wallet:', err);
        } finally {
            setIsConnecting(false);
        }
    };

    // Switch network
    const switchNetwork = async (networkName = DEFAULT_NETWORK) => {
        if (!window.ethereum) {
            setError('Please install MetaMask');
            return;
        }

        try {
            const networkConfig = getNetworkConfig(networkName);
            await window.ethereum.request({
                method: 'wallet_switchEthereumChain',
                params: [{ chainId: networkConfig.chainId }],
            });
        } catch (err) {
            // If the network doesn't exist, add it
            if (err.code === 4902) {
                try {
                    const networkConfig = getNetworkConfig(networkName);
                    await window.ethereum.request({
                        method: 'wallet_addEthereumChain',
                        params: [networkConfig],
                    });
                } catch (addErr) {
                    setError(addErr.message);
                    console.error('Error adding network:', addErr);
                }
            } else {
                setError(err.message);
                console.error('Error switching network:', err);
            }
        }
    };

    // Disconnect wallet
    const disconnectWallet = () => {
        setAccount(null);
        setSigner(null);
    };

    return {
        account,
        provider,
        signer,
        contracts,
        chainId,
        isConnecting,
        error,
        connectWallet,
        disconnectWallet,
        switchNetwork
    };
}; 