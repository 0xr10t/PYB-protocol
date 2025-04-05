# Perpetual Yield Bonds (PYB) Protocol

A DeFi protocol that issues "perpetual yield bonds", where users lock assets (like ETH, stETH, DAI), and the interest is streamed to bondholders. These bonds are NFTs that can be traded on secondary markets.

## Features

- Bond issuance with NFT representation
- Real-time yield streaming
- AI-powered strategy optimization
- Secondary market trading
- Protocol treasury management
- Emergency reserves

## Smart Contracts

1. **BondFactory**: Manages bond creation and deployment
2. **BondToken**: ERC-721 implementation for bond NFTs
3. **YieldDistribution**: Handles yield streaming and distribution
4. **StrategyManager**: Manages lending protocol interactions
5. **ProtocolTreasury**: Manages protocol fees and reserves

## Setup

1. Install dependencies:
```bash
forge install
```

2. Set up environment variables:
```bash
export PRIVATE_KEY=your_private_key
export SEPOLIA_RPC_URL=your_sepolia_rpc_url
export ETHERSCAN_API_KEY=your_etherscan_api_key
```

3. Deploy contracts:
```bash
forge script script/Deploy.s.sol:DeployScript --rpc-url $SEPOLIA_RPC_URL --broadcast
```

## Development

- Solidity version: 0.8.19
- Framework: Foundry
- Network: Sepolia testnet

## Testing

Run tests:
```bash
forge test
```

## License

MIT
