import { CONTRACT_ADDRESSES as DEPLOYED_ADDRESSES } from '../contracts/addresses';

// Network configurations
export const NETWORKS = {
    // Add your network configurations here
    // Example for Ethereum mainnet:
    mainnet: {
        chainId: '0x1',
        chainName: 'Ethereum Mainnet',
        rpcUrls: ['https://mainnet.infura.io/v3/YOUR-PROJECT-ID'],
        nativeCurrency: {
            name: 'Ether',
            symbol: 'ETH',
            decimals: 18
        },
        blockExplorerUrls: ['https://etherscan.io']
    },
    // Example for a testnet:
    sepolia: {
        chainId: '0xaa36a7',
        chainName: 'Sepolia Testnet',
        rpcUrls: ['https://sepolia.infura.io/v3/YOUR-PROJECT-ID'],
        nativeCurrency: {
            name: 'Sepolia Ether',
            symbol: 'ETH',
            decimals: 18
        },
        blockExplorerUrls: ['https://sepolia.etherscan.io']
    }
};

// Contract addresses from addresses.js
export const CONTRACT_ADDRESSES = {
    // Mainnet addresses
    mainnet: {
        BOND_FACTORY: DEPLOYED_ADDRESSES.BondFactory,
        PROTOCOL_TREASURY: DEPLOYED_ADDRESSES.ProtocolTreasury,
        STRATEGY_MANAGER: DEPLOYED_ADDRESSES.StrategyManager,
        YIELD_DISTRIBUTION: DEPLOYED_ADDRESSES.YieldDistribution,
        BOND_TOKEN: DEPLOYED_ADDRESSES.BondToken
    },
    // Testnet addresses (using same addresses for now)
    sepolia: {
        BOND_FACTORY: DEPLOYED_ADDRESSES.BondFactory,
        PROTOCOL_TREASURY: DEPLOYED_ADDRESSES.ProtocolTreasury,
        STRATEGY_MANAGER: DEPLOYED_ADDRESSES.StrategyManager,
        YIELD_DISTRIBUTION: DEPLOYED_ADDRESSES.YieldDistribution,
        BOND_TOKEN: DEPLOYED_ADDRESSES.BondToken
    }
};

// Default network to use
export const DEFAULT_NETWORK = 'sepolia';

// RPC URLs for different networks
export const RPC_URLS = {
    mainnet: 'https://mainnet.infura.io/v3/YOUR-PROJECT-ID',
    sepolia: 'https://sepolia.infura.io/v3/YOUR-PROJECT-ID'
};

// Helper function to get contract address based on network
export const getContractAddress = (contractName, network = DEFAULT_NETWORK) => {
    return CONTRACT_ADDRESSES[network][contractName];
};

// Helper function to get network configuration
export const getNetworkConfig = (network = DEFAULT_NETWORK) => {
    return NETWORKS[network];
}; 