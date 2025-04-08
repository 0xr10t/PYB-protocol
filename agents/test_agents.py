import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from web3 import Web3
import time

from yield_strategy_agent import YieldStrategyAgent
from risk_management_agent import RiskManagementAgent
from rebalancing_agent import RebalancingAgent
from price_oracle_agent import PriceOracleAgent
from treasury_management_agent import TreasuryManagementAgent

def generate_mock_data():
    """Generate mock data for testing"""
    # Mock token addresses
    eth_address = '0x1234567890123456789012345678901234567890'
    wbtc_address = '0x2468135790246813579024681357902468135790'
    usdc_address = '0x3579246813579024681357902468135790246813'
    
    # Mock token prices with historical data
    token_prices = {
        eth_address: [(2000.0, int(time.time()))],  # $2000 per ETH
        wbtc_address: [(30000.0, int(time.time()))],  # $30000 per WBTC
        usdc_address: [(1.0, int(time.time()))]  # $1 per USDC
    }
    
    # Mock strategy data
    strategies = {
        eth_address: {
            0: {
                'deposited_amount': Web3.to_wei(100, 'ether'),  # 100 ETH
                'optimal_amount': Web3.to_wei(120, 'ether'),  # 120 ETH
                'last_rebalance': int(time.time()) - 86400  # 24 hours ago
            },
            1: {
                'deposited_amount': Web3.to_wei(50, 'ether'),  # 50 ETH
                'optimal_amount': Web3.to_wei(45, 'ether'),  # 45 ETH
                'last_rebalance': int(time.time()) - 86400  # 24 hours ago
            }
        },
        wbtc_address: {
            0: {
                'deposited_amount': Web3.to_wei(10, 'ether'),  # 10 WBTC
                'optimal_amount': Web3.to_wei(12, 'ether'),  # 12 WBTC
                'last_rebalance': int(time.time()) - 86400  # 24 hours ago
            },
            1: {
                'deposited_amount': Web3.to_wei(5, 'ether'),  # 5 WBTC
                'optimal_amount': Web3.to_wei(4, 'ether'),  # 4 WBTC
                'last_rebalance': int(time.time()) - 86400  # 24 hours ago
            }
        }
    }
    
    # Mock treasury data
    treasury_metrics = {
        'total_assets': 750000.0,  # $750,000 total assets
        'total_liabilities': 225000.0,  # $225,000 total liabilities
        'reserve_ratio': 0.7  # 70% reserve ratio
    }
    
    # Mock historical data for rebalancing
    historical_data = []
    for token, series_data in strategies.items():
        for series_id, data in series_data.items():
            historical_data.append({
                'token': token,
                'series_id': series_id,
                'deposited_amount': data['deposited_amount'],
                'last_rebalance': data['last_rebalance']
            })
    
    # Mock features for rebalancing prediction
    features = np.array([
        [1.0, 100.0, 120.0, 0.2],  # ETH series 0
        [1.0, 50.0, 45.0, -0.1],   # ETH series 1
        [1.0, 10.0, 12.0, 0.2],    # WBTC series 0
        [1.0, 5.0, 4.0, -0.2]      # WBTC series 1
    ])
    
    return {
        'token_prices': token_prices,
        'prices': token_prices,  # Add prices key for PriceOracleAgent
        'strategies': strategies,
        'treasury_metrics': treasury_metrics,
        'historical_data': historical_data,
        'features': features,
        'rebalance_prediction': True,
        'rebalance_result': True,
        'reserve_optimization': True
    }

def test_yield_strategy_agent(agent, mock_data):
    """Test the yield strategy agent"""
    print("\nTesting Yield Strategy Agent...")
    
    # Get current market data
    market_data = []
    for token, series_data in mock_data['strategies'].items():
        for series_id, data in series_data.items():
            market_data.append({
                'token': token,
                'series_id': series_id,  # series_id is already an integer
                'lending_protocol': data['lending_protocol'],
                'deposited_amount': data['deposited_amount'][-1],
                'last_rebalance': data['last_rebalance'][-1],
                'optimal_amount': data['optimal_amount'][-1]
            })
    
    market_data = pd.DataFrame(market_data)
    
    # Predict optimal strategy
    best_token, best_series, expected_amount = agent.predict_optimal_strategy(market_data)
    print(f"Best Token: {best_token}")
    print(f"Best Series: {best_series}")
    print(f"Expected Amount: {expected_amount:,.2f}")
    
    # Optimize portfolio
    result = agent.optimize_portfolio()
    print("\nPortfolio Optimization Result:")
    print(f"Token: {result['token']}")
    print(f"Series ID: {result['series_id']}")
    print(f"Amount: {result['amount']:,.2f}")
    print(f"Transaction Hash: {result['tx_hash']}")

def test_risk_management_agent(agent, mock_data):
    """Test the risk management agent"""
    print("\nTesting Risk Management Agent...")
    
    # Get current portfolio state
    portfolio_state = {}
    for token, series_data in mock_data['strategies'].items():
        for series_id, data in series_data.items():
            portfolio_state[f"{token}_{series_id}"] = {
                'tvl': data['deposited_amount'][-1],
                'risk_score': 0.5  # Mock risk score
            }
    
    # Assess portfolio risk
    risk_assessment = agent.assess_portfolio_risk(portfolio_state)
    print("Risk Assessment:")
    print(f"Overall Risk Score: {risk_assessment['overall_risk_score']:.2f}")
    print("Risk Factors:")
    for factor, score in risk_assessment['risk_factors'].items():
        print(f"- {factor}: {score:.2f}")
    
    # Get risk mitigation recommendations
    recommendations = agent.get_risk_mitigation_recommendations(risk_assessment)
    print("\nRisk Mitigation Recommendations:")
    for rec in recommendations:
        print(f"- {rec}")

def test_rebalancing_agent(agent, mock_data):
    """Test the rebalancing agent"""
    print("\nTesting Rebalancing Agent...")
    
    # Get current weights
    current_weights = {
        'ETH_Staking': 0.4,
        'WBTC_Yield': 0.3,
        'USDC_Lending': 0.3
    }
    
    # Monitor and rebalance
    result = agent.monitor_and_rebalance()
    print("Rebalancing Result:")
    print(f"Rebalancing Needed: {result['rebalanced']}")
    if result['rebalanced']:
        print("Old Weights:")
        for strategy, weight in result['old_weights'].items():
            print(f"- {strategy}: {weight:.2%}")
        print("New Weights:")
        for strategy, weight in result['new_weights'].items():
            print(f"- {strategy}: {weight:.2%}")

def test_price_oracle_agent(agent, mock_data):
    """Test the price oracle agent"""
    print("\nTesting Price Oracle Agent...")
    
    # Monitor prices for each token
    tokens = ['ETH', 'WBTC', 'USDC']
    results = agent.monitor_prices(tokens)
    
    print("Price Predictions:")
    for token, data in results.items():
        print(f"\n{token}:")
        print(f"Current Price: ${data['current_price']:,.2f}")
        print(f"Predicted Price: ${data['predicted_price']:,.2f}")
        print(f"24h Change: {data['price_change_24h']:.2%}")
        print(f"Prediction Change: {data['prediction_change']:.2%}")

def test_treasury_management_agent(agent, mock_data):
    """Test the treasury management agent"""
    print("\nTesting Treasury Management Agent...")
    
    # Manage treasury
    result = agent.manage_treasury()
    
    print("Treasury Management Result:")
    print(f"Action Probability: {result['action_probability']:.2%}")
    print("\nMetrics:")
    for metric, value in result['metrics'].items():
        print(f"- {metric}: {value:,.2f}")
    print("\nActions Taken:")
    for action in result['actions_taken']:
        print(f"- {action}")

def main():
    # Initialize Web3
    w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
    
    # Contract addresses
    contract_addresses = {
        'strategy_manager': '0x1234567890123456789012345678901234567890',
        'treasury': '0x2468135790246813579024681357902468135790',
        'price_oracle': '0x3579246813579024681357902468135790246813'
    }
    
    # Generate mock data
    mock_data = generate_mock_data()
    
    # Test Yield Strategy Agent
    print("\nTesting Yield Strategy Agent...")
    yield_agent = YieldStrategyAgent(w3.provider.endpoint_uri, contract_addresses, is_testing=True)
    yield_agent.mock_data = mock_data
    
    best_token = yield_agent.identify_best_token()
    print(f"Best token: {best_token}")
    
    expected_amount = yield_agent.calculate_expected_amount(best_token)
    print(f"Expected amount: {expected_amount}")
    
    # Test Risk Management Agent
    print("\nTesting Risk Management Agent...")
    risk_agent = RiskManagementAgent(w3.provider.endpoint_uri, contract_addresses, is_testing=True)
    risk_agent.mock_data = mock_data
    
    risk_score = risk_agent.assess_portfolio_risk()
    print(f"Portfolio risk score: {risk_score}")
    
    recommendations = risk_agent.get_risk_mitigation_recommendations()
    print("Risk mitigation recommendations:")
    for rec in recommendations:
        print(f"- {rec}")
    
    # Test Rebalancing Agent
    print("\nTesting Rebalancing Agent...")
    rebalance_agent = RebalancingAgent(w3.provider.endpoint_uri, contract_addresses, is_testing=True)
    rebalance_agent.mock_data = mock_data
    
    result = rebalance_agent.monitor_and_rebalance()
    print(f"Rebalancing needed: {result}")
    
    # Test Price Oracle Agent
    print("\nTesting Price Oracle Agent...")
    price_agent = PriceOracleAgent(w3.provider.endpoint_uri, contract_addresses, is_testing=True)
    price_agent.mock_data = mock_data
    
    # Use full token addresses
    tokens = [
        '0x1234567890123456789012345678901234567890',  # ETH
        '0x2468135790246813579024681357902468135790',  # WBTC
        '0x3579246813579024681357902468135790246813'   # USDC
    ]
    
    for token in tokens:
        price = price_agent.get_token_price(token)
        print(f"Token {token}: ${price[0]:,.2f}")
    
    # Test Treasury Management Agent
    print("\nTesting Treasury Management Agent...")
    treasury_agent = TreasuryManagementAgent(w3.provider.endpoint_uri, contract_addresses, is_testing=True)
    treasury_agent.mock_data = mock_data
    
    metrics = treasury_agent.get_treasury_metrics()
    print("Treasury metrics:", metrics)

if __name__ == "__main__":
    main() 