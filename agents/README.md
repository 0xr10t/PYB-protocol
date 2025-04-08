# PYB Protocol AI Agents

This directory contains the AI agents that power the PYB Protocol's autonomous operations. The agents use machine learning and data analysis to optimize yield strategies, manage risks, rebalance portfolios, monitor prices, and manage treasury operations.

## Agents

### 1. Yield Strategy Agent
- Optimizes yield strategies based on market conditions
- Uses neural networks to predict optimal strategy allocations
- Monitors and adjusts strategy weights based on performance

### 2. Risk Management Agent
- Assesses and monitors protocol risks
- Uses isolation forest for anomaly detection
- Implements risk mitigation strategies
- Maintains protocol safety parameters

### 3. Rebalancing Agent
- Determines optimal times for rebalancing investments
- Uses LSTM networks for prediction
- Monitors portfolio drift and executes rebalancing
- Maintains target allocation weights

### 4. Price Oracle Agent
- Predicts price movements to inform rebalancing actions
- Integrates with Chainlink price feeds
- Uses technical indicators and machine learning
- Provides price predictions and market insights

### 5. Treasury Management Agent
- Manages protocol reserves and optimizes fee collection
- Maintains optimal reserve ratios
- Adjusts protocol fees based on market conditions
- Manages treasury operations

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy the environment template and fill in your values:
```bash
cp .env.example .env
```

3. Update the `.env` file with your:
- Web3 provider URL
- Contract addresses
- Private key for transaction signing
- Chainlink price feed addresses
- Agent configuration parameters

## Usage

Run the agents:
```bash
python run_agents.py
```

The agents will run continuously, executing their respective operations every 5 minutes. Results will be printed to the console.

## Configuration

### Environment Variables

- `WEB3_PROVIDER`: Web3 provider URL (e.g., Infura)
- `STRATEGY_MANAGER_ADDRESS`: Address of the StrategyManager contract
- `PROTOCOL_TREASURY_ADDRESS`: Address of the ProtocolTreasury contract
- `YIELD_DISTRIBUTION_ADDRESS`: Address of the YieldDistribution contract
- `BOND_FACTORY_ADDRESS`: Address of the BondFactory contract
- `PRIVATE_KEY`: Private key for transaction signing
- `ETH_USD_PRICE_FEED`: Chainlink ETH/USD price feed address
- `WBTC_USD_PRICE_FEED`: Chainlink WBTC/USD price feed address
- `USDC_USD_PRICE_FEED`: Chainlink USDC/USD price feed address

### Agent Parameters

- `MIN_RESERVE_RATIO`: Minimum treasury reserve ratio (default: 0.1)
- `MAX_RESERVE_RATIO`: Maximum treasury reserve ratio (default: 0.3)
- `REBALANCE_THRESHOLD`: Threshold for portfolio rebalancing (default: 0.1)
- `MIN_COLLATERAL_RATIO`: Minimum collateral ratio (default: 1.5)
- `MAX_CONCENTRATION`: Maximum strategy concentration (default: 0.4)

## Architecture

The agents are coordinated by an `Orchestrator` class that:
1. Initializes all agents
2. Manages agent states
3. Runs agent operations in sequence
4. Handles errors and retries
5. Provides access to agent results

Each agent:
1. Inherits from `BaseAgent`
2. Implements its specific functionality
3. Uses machine learning models for predictions
4. Interacts with smart contracts
5. Maintains its own state

## Machine Learning Models

The agents use various machine learning models:

1. Yield Strategy Agent:
   - Dense neural network for yield prediction
   - Input: Market data, strategy metrics
   - Output: Optimal strategy weights

2. Risk Management Agent:
   - Isolation Forest for anomaly detection
   - Input: Protocol metrics
   - Output: Risk scores and anomalies

3. Rebalancing Agent:
   - LSTM network for rebalancing prediction
   - Input: Historical price and performance data
   - Output: Rebalancing decisions

4. Price Oracle Agent:
   - LSTM network for price prediction
   - Input: Historical price data and technical indicators
   - Output: Price predictions

5. Treasury Management Agent:
   - Dense neural network for treasury management
   - Input: Treasury metrics
   - Output: Treasury action decisions

## Security Considerations

1. Private Key Management:
   - Store private keys securely
   - Use environment variables
   - Never commit private keys to version control

2. Transaction Signing:
   - All transactions are signed with the provided private key
   - Gas prices are automatically calculated
   - Transaction parameters are validated

3. Error Handling:
   - All operations are wrapped in try-except blocks
   - Errors are logged and handled gracefully
   - Failed operations are retried after a delay

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 