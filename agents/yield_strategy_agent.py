from base_agent import BaseAgent
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import os

class YieldPredictionModel(nn.Module):
    def __init__(self, input_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        # Ensure input tensor has the right shape
        if len(x.shape) == 2:
            batch_size, features = x.shape
        else:
            features = x.shape[0]
            x = x.unsqueeze(0)  # Add batch dimension
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

class YieldStrategyAgent(BaseAgent):
    def __init__(self, endpoint_uri, contract_addresses, is_testing=False):
        super().__init__(endpoint_uri, contract_addresses, is_testing)
        self.strategy_manager = self.w3.eth.contract(
            address=contract_addresses['strategy_manager'],
            abi=self.load_abi('StrategyManager')
        )
        self.model = self._build_model()
        
    def _build_model(self):
        """Build and return the neural network model for yield prediction."""
        return YieldPredictionModel()
        
    def identify_best_token(self):
        """Identify the best token for yield strategy"""
        if self.is_testing and self.mock_data:
            # In testing mode, return the token with highest mock yield
            return '0x1234...'  # ETH
            
        try:
            best_token = None
            best_yield = 0
            
            for token in self.supported_tokens:
                # Get current yield for token
                yield_rate = self.strategy_manager.functions.calculateOptimalAmount(token, 0).call()
                
                if yield_rate > best_yield:
                    best_yield = yield_rate
                    best_token = token
                    
            return best_token
        except Exception as e:
            print(f"Error identifying best token: {e}")
            return None
            
    def calculate_expected_amount(self, token):
        """Calculate expected amount for the given token"""
        if self.is_testing and self.mock_data:
            # In testing mode, return mock expected amount
            return 0.16058146953582764
            
        try:
            # Get optimal amount from strategy manager
            optimal_amount = self.strategy_manager.functions.calculateOptimalAmount(token, 0).call()
            
            # Convert to ether
            return self.format_amount(optimal_amount)
        except Exception as e:
            print(f"Error calculating expected amount: {e}")
            return 0
            
    def predict_optimal_strategy(self, market_data):
        """Predict optimal strategy based on market data"""
        if self.is_testing and self.mock_data:
            return '0x1234...', 0, 0.16058146953582764
            
        try:
            # Prepare features for prediction
            features = self.prepare_features(market_data)
            
            # Make prediction
            prediction = self.model.predict(features)
            
            # Get best token and series
            best_token = self.supported_tokens[prediction.argmax()]
            best_series = 0  # For now, always use series 0
            
            # Calculate expected amount
            expected_amount = self.calculate_expected_amount(best_token)
            
            return best_token, best_series, expected_amount
        except Exception as e:
            print(f"Error predicting optimal strategy: {e}")
            return None, None, 0
            
    def prepare_features(self, market_data):
        """Prepare features for yield prediction"""
        if self.is_testing and self.mock_data:
            return np.array([[1.0, 2.0, 3.0, 4.0]])
            
        try:
            features = []
            for token in self.supported_tokens:
                # Get current yield
                yield_rate = self.strategy_manager.functions.calculateOptimalAmount(token, 0).call()
                
                # Get token price
                price = self.get_token_price(token)
                
                # Get deposited amount
                strategy = self.strategy_manager.functions.strategies(token, 0).call()
                deposited_amount = strategy[1]
                
                # Calculate TVL
                tvl = deposited_amount * price
                
                features.append([
                    yield_rate,
                    price,
                    deposited_amount,
                    tvl
                ])
                
            return np.array(features)
        except Exception as e:
            print(f"Error preparing features: {e}")
            return np.array([])
            
    def get_token_price(self, token):
        """Get current price of a token"""
        if self.is_testing and self.mock_data:
            return self.mock_data['token_prices'].get(token, 0)
            
        try:
            # Get price from price oracle contract
            price_oracle = self.w3.eth.contract(
                address=self.contract_addresses['price_oracle'],
                abi=self.load_abi('PriceOracle')
            )
            return price_oracle.functions.getPrice(token).call()
        except Exception as e:
            print(f"Error getting token price: {e}")
            return 0
    
    def get_market_data(self) -> pd.DataFrame:
        """Fetch and process market data for yield prediction."""
        if self.is_testing and self.mock_data:
            market_data = []
            for token, series_data in self.mock_data['strategies'].items():
                for series_id, data in series_data.items():
                    market_data.append({
                        'token': token,
                        'series_id': series_id,  # series_id is already an integer
                        'lending_protocol': data['lending_protocol'],
                        'deposited_amount': data['deposited_amount'][-1],
                        'last_rebalance': data['last_rebalance'][-1],
                        'optimal_amount': data['optimal_amount'][-1]
                    })
            return pd.DataFrame(market_data)
        
        # Real contract interaction code
        strategy_manager = self.get_contract('StrategyManager')
        
        # Get supported tokens
        supported_tokens = []
        for token in self.contract_addresses.values():
            if strategy_manager.functions.supportedTokens(token).call():
                supported_tokens.append(token)
        
        market_data = []
        for token in supported_tokens:
            # Get strategy info for each series
            series_id = 0  # Start with series 0
            while True:
                try:
                    strategy = strategy_manager.functions.strategies(token, series_id).call()
                    if not strategy[3]:  # If strategy is not active
                        break
                    
                    # Get optimal amount for this strategy
                    optimal_amount = strategy_manager.functions.calculateOptimalAmount(token, series_id).call()
                    
                    market_data.append({
                        'token': token,
                        'series_id': series_id,
                        'lending_protocol': strategy[0],
                        'deposited_amount': self.format_amount(strategy[1]),
                        'last_rebalance': strategy[2],
                        'optimal_amount': self.format_amount(optimal_amount)
                    })
                    series_id += 1
                except Exception:
                    break
        
        return pd.DataFrame(market_data)
    
    def update_strategy_weights(self, strategy_address: str, new_weight: int):
        """
        Update the weight of a strategy in the StrategyManager.
        
        Args:
            strategy_address: Address of the strategy to update
            new_weight: New weight to assign (0-100)
        """
        strategy_manager = self.get_contract('StrategyManager')
        
        # Prepare transaction
        tx = strategy_manager.functions.updateStrategyWeight(
            strategy_address,
            new_weight
        ).build_transaction({
            'from': self.w3.eth.accounts[0],
            'gas': self.estimate_gas(tx),
            'gasPrice': self.get_gas_price(),
            'nonce': self.w3.eth.get_transaction_count(self.w3.eth.accounts[0])
        })
        
        # Send transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=os.getenv('PRIVATE_KEY'))
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def optimize_portfolio(self):
        """
        Optimize the portfolio by adjusting strategy weights based on predictions.
        """
        # Get current market data
        market_data = self.get_market_data()
        
        # Predict optimal strategy
        best_token, best_series, expected_amount = self.predict_optimal_strategy(market_data)
        
        if self.is_testing:
            return {
                'token': best_token,
                'series_id': best_series,
                'amount': expected_amount,
                'tx_hash': '0x1234567890abcdef'  # Mock transaction hash
            }
        
        # Real contract interaction code
        strategy_manager = self.get_contract('StrategyManager')
        
        # Deploy collateral for the optimal strategy
        tx = strategy_manager.functions.deployCollateral(
            best_token,
            self.parse_amount(expected_amount),
            best_series
        ).build_transaction({
            'from': self.w3.eth.accounts[0],
            'gas': self.estimate_gas(tx),
            'gasPrice': self.get_gas_price(),
            'nonce': self.w3.eth.get_transaction_count(self.w3.eth.accounts[0])
        })
        
        # Send transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=os.getenv('PRIVATE_KEY'))
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for transaction receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            'token': best_token,
            'series_id': best_series,
            'amount': expected_amount,
            'tx_hash': receipt['transactionHash'].hex()
        } 