from base_agent import BaseAgent
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import os
import time
from web3 import Web3
import json

class RebalancingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
        
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            return self.forward(x_tensor).numpy()

class RebalancingAgent(BaseAgent):
    def __init__(self, endpoint_uri, contract_addresses, is_testing=False):
        super().__init__(endpoint_uri, contract_addresses, is_testing)
        self.strategy_manager = self.w3.eth.contract(
            address=contract_addresses['strategy_manager'],
            abi=self.load_abi('StrategyManager')
        )
        self.rebalance_interval = 86400  # 24 hours in seconds
        self.min_rebalance_amount = Web3.to_wei(0.1, 'ether')  # 0.1 ETH minimum
        self.model = self._build_model()
        
    def _build_model(self):
        """Build and return the neural network model for rebalancing prediction."""
        return RebalancingModel()
        
    def get_historical_data(self, days=30):
        """Get historical data for all active strategies"""
        if self.is_testing and self.mock_data:
            return self.mock_data['historical_data']
            
        try:
            # Get all supported tokens and their strategies
            active_strategies = []
            for token in self.supported_tokens:
                for series_id in range(10):  # Check first 10 series
                    strategy = self.strategy_manager.functions.strategies(token, series_id).call()
                    if strategy[4]:  # strategy.active
                        active_strategies.append({
                            'token': token,
                            'series_id': series_id,
                            'deposited_amount': strategy[1],
                            'last_rebalance': strategy[2]
                        })
            return active_strategies
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return []
            
    def prepare_features(self, historical_data):
        """Prepare features for the LSTM model"""
        if self.is_testing and self.mock_data:
            return self.mock_data['features']
            
        features = []
        for strategy in historical_data:
            # Calculate time since last rebalance
            time_since_rebalance = (int(time.time()) - strategy['last_rebalance']) / self.rebalance_interval
            
            # Get current deposited amount
            current_amount = strategy['deposited_amount']
            
            # Calculate optimal amount from contract
            optimal_amount = self.strategy_manager.functions.calculateOptimalAmount(
                strategy['token'],
                strategy['series_id']
            ).call()
            
            # Calculate deviation from optimal
            deviation = (optimal_amount - current_amount) / current_amount if current_amount > 0 else 0
            
            features.append([
                time_since_rebalance,
                current_amount,
                optimal_amount,
                deviation
            ])
            
        return np.array(features)
        
    def predict_rebalance_needed(self, features):
        """Predict if rebalancing is needed"""
        if self.is_testing and self.mock_data:
            return self.mock_data['rebalance_prediction']
            
        if len(features) == 0:
            return False
            
        # Reshape features for LSTM input
        features = features.reshape((1, 1, 4))
        
        # Make prediction
        prediction = self.model.predict(features)
        return prediction[0][0] > 0.5
        
    def execute_rebalance(self, target_weights: Dict[str, float]):
        """
        Execute the rebalancing by updating strategy weights.
        
        Args:
            target_weights: Target weights for each strategy
        """
        strategy_manager = self.get_contract('StrategyManager')
        
        for strategy, weight in target_weights.items():
            # Convert weight to percentage (0-100)
            weight_percentage = int(weight * 100)
            
            # Update strategy weight
            tx = strategy_manager.functions.updateStrategyWeight(
                strategy,
                weight_percentage
            ).build_transaction({
                'from': self.w3.eth.accounts[0],
                'gas': self.estimate_gas(tx),
                'gasPrice': self.get_gas_price(),
                'nonce': self.w3.eth.get_transaction_count(self.w3.eth.accounts[0])
            })
            
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=os.getenv('PRIVATE_KEY'))
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def monitor_and_rebalance(self):
        """
        Main function to monitor and execute rebalancing when needed.
        """
        if self.is_testing and self.mock_data:
            return self.mock_data.get('rebalance_result', False)
            
        try:
            # Get historical data for the last day
            historical_data = self.get_historical_data(days=1)
            
            if not historical_data:
                return False
                
            # Get current weights from historical data
            current_weights = {}
            for strategy in historical_data:
                total_value = sum(s['deposited_amount'] for s in historical_data)
                if total_value > 0:
                    current_weights[f"{strategy['token']}_{strategy['series_id']}"] = strategy['deposited_amount'] / total_value
                else:
                    current_weights[f"{strategy['token']}_{strategy['series_id']}"] = 0
                    
            # Calculate target weights
            target_weights = self.calculate_target_weights(current_weights)
            
            # Check if rebalancing is needed
            if self.check_rebalance_needed(current_weights, target_weights):
                # Execute rebalancing
                self.execute_rebalance(target_weights)
                return True
                
            return False
            
        except Exception as e:
            print(f"Error in monitor_and_rebalance: {e}")
            return False
    
    def calculate_target_weights(self, current_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate target weights for strategies based on performance.
        
        Args:
            current_weights: Current strategy weights
            
        Returns:
            Dictionary of target weights
        """
        strategy_manager = self.get_contract('StrategyManager')
        strategies = list(current_weights.keys())
        
        # Get performance metrics
        performance_scores = {}
        for strategy in strategies:
            apy = self.format_amount(strategy_manager.functions.getStrategyAPY(strategy).call())
            risk_score = strategy_manager.functions.getStrategyRiskScore(strategy).call()
            
            # Calculate performance score (APY / risk)
            performance_scores[strategy] = apy / risk_score if risk_score > 0 else 0
        
        # Calculate total score
        total_score = sum(performance_scores.values())
        
        # Calculate target weights
        target_weights = {}
        for strategy in strategies:
            target_weights[strategy] = performance_scores[strategy] / total_score if total_score > 0 else 1.0 / len(strategies)
        
        return target_weights
    
    def check_rebalance_needed(self, current_weights: Dict[str, float], target_weights: Dict[str, float]) -> bool:
        """
        Check if rebalancing is needed based on weight deviations.
        
        Args:
            current_weights: Current strategy weights
            target_weights: Target strategy weights
            
        Returns:
            True if rebalancing is needed
        """
        for strategy in current_weights:
            deviation = abs(current_weights[strategy] - target_weights[strategy])
            if deviation > self.rebalance_threshold:
                return True
        
        return False 