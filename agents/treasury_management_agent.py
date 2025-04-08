from base_agent import BaseAgent
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import os
from datetime import datetime, timedelta

class TreasuryManagementModel(nn.Module):
    def __init__(self, input_size=8):
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

class TreasuryManagementAgent(BaseAgent):
    def __init__(self, endpoint_uri, contract_addresses, is_testing=False):
        super().__init__(endpoint_uri, contract_addresses, is_testing)
        self.treasury = self.w3.eth.contract(
            address=contract_addresses['treasury'],
            abi=self.load_abi('ProtocolTreasury')
        )
        self.min_reserve_ratio = 0.1  # 10% minimum reserve ratio
        self.max_reserve_ratio = 0.3  # 30% maximum reserve ratio
        self.target_reserve_ratio = 0.2  # 20% target reserve ratio
        
    def get_treasury_metrics(self):
        """Get current treasury metrics"""
        if self.is_testing and self.mock_data:
            return self.mock_data['treasury_metrics']
            
        try:
            metrics = {}
            total_assets = 0
            total_liabilities = 0
            
            # Get reserves for each supported token
            for token in self.supported_tokens:
                reserves = self.treasury.functions.getReserves(token).call()
                total_fees, emergency_reserve, last_update = reserves
                
                # Get token price
                price = self.get_token_price(token)
                
                # Calculate USD values
                fees_usd = total_fees * price
                reserve_usd = emergency_reserve * price
                
                total_assets += fees_usd
                total_liabilities += reserve_usd
                
                metrics[token] = {
                    'total_fees': total_fees,
                    'emergency_reserve': emergency_reserve,
                    'last_update': last_update,
                    'fees_usd': fees_usd,
                    'reserve_usd': reserve_usd
                }
            
            # Calculate overall metrics
            metrics['total_assets'] = total_assets
            metrics['total_liabilities'] = total_liabilities
            metrics['reserve_ratio'] = total_liabilities / total_assets if total_assets > 0 else 0
            
            return metrics
        except Exception as e:
            print(f"Error getting treasury metrics: {e}")
            return {
                'total_assets': 0,
                'total_liabilities': 0,
                'reserve_ratio': 0
            }
            
    def optimize_reserves(self):
        """Optimize treasury reserves"""
        if self.is_testing and self.mock_data:
            return self.mock_data['reserve_optimization']
            
        try:
            metrics = self.get_treasury_metrics()
            current_ratio = metrics['reserve_ratio']
            
            # Check if reserve ratio needs adjustment
            if current_ratio < self.min_reserve_ratio:
                # Need to increase reserves
                target_ratio = self.target_reserve_ratio
            elif current_ratio > self.max_reserve_ratio:
                # Need to decrease reserves
                target_ratio = self.target_reserve_ratio
            else:
                # Reserve ratio is within acceptable range
                return False
                
            # Calculate required changes
            target_reserves = metrics['total_assets'] * target_ratio
            current_reserves = metrics['total_liabilities']
            reserve_change = target_reserves - current_reserves
            
            if abs(reserve_change) < 100:  # Minimum $100 change required
                return False
                
            # Update emergency reserve ratio in contract
            new_ratio_bps = int(target_ratio * 10000)  # Convert to basis points
            tx = self.treasury.functions.setEmergencyReserveRatio(new_ratio_bps).build_transaction({
                'from': self.w3.eth.accounts[0],
                'gas': 2000000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.w3.eth.accounts[0])
            })
            
            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction receipt
            self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return True
        except Exception as e:
            print(f"Error optimizing reserves: {e}")
            return False
            
    def get_token_price(self, token_address):
        """Get current price of a token in USD"""
        if self.is_testing and self.mock_data:
            return self.mock_data['token_prices'].get(token_address, 0)
            
        try:
            # Get price from price oracle contract
            price_oracle = self.w3.eth.contract(
                address=self.contract_addresses['price_oracle'],
                abi=self.load_abi('PriceOracle')
            )
            return price_oracle.functions.getPrice(token_address).call()
        except Exception as e:
            print(f"Error getting token price: {e}")
            return 0
    
    def calculate_fee_income(self) -> float:
        """Calculate the fee income over the last 30 days."""
        treasury = self.get_contract('ProtocolTreasury')
        
        # Get fee history for the last 30 days
        fee_history = treasury.functions.getFeeHistory(30).call()
        
        # Calculate total fee income
        total_fees = sum(self.format_amount(fee) for fee in fee_history)
        
        return total_fees
    
    def calculate_debt_ratio(self) -> float:
        """Calculate the current debt ratio."""
        treasury = self.get_contract('ProtocolTreasury')
        
        total_debt = self.format_amount(treasury.functions.getTotalDebt().call())
        total_tvl = self.format_amount(treasury.functions.getTotalTVL().call())
        
        if total_tvl == 0:
            return 0.0
        
        return total_debt / total_tvl
    
    def prepare_features(self, metrics: Dict[str, float]) -> np.ndarray:
        """
        Prepare features for the treasury management model.
        
        Args:
            metrics: Dictionary of treasury metrics
        """
        features = np.array([
            metrics['total_reserves'],
            metrics['total_debt'],
            metrics['protocol_fees'],
            metrics['reserve_ratio'],
            metrics['fee_income'],
            metrics['debt_ratio'],
            self.min_reserve_ratio,
            self.max_reserve_ratio
        ]).reshape(1, -1)
        
        return features
    
    def predict_treasury_action(self, features: np.ndarray) -> float:
        """
        Predict whether treasury action is needed.
        
        Args:
            features: Prepared feature array
            
        Returns:
            Probability that treasury action is needed
        """
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Convert to PyTorch tensor
        features_tensor = torch.FloatTensor(features_scaled)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Predict action probability
        with torch.no_grad():
            prediction = self.model(features_tensor)
        
        return float(prediction[0][0])
    
    def optimize_fee_collection(self):
        """Optimize fee collection strategy."""
        treasury = self.get_contract('ProtocolTreasury')
        
        # Get current fee settings
        current_fee = treasury.functions.getProtocolFee().call()
        
        # Calculate optimal fee based on market conditions
        optimal_fee = self.calculate_optimal_fee()
        
        if abs(current_fee - optimal_fee) > 0.001:  # 0.1% difference threshold
            # Update protocol fee
            tx = treasury.functions.setProtocolFee(
                int(optimal_fee * 10000)  # Convert to basis points
            ).build_transaction({
                'from': self.w3.eth.accounts[0],
                'gas': self.estimate_gas(tx),
                'gasPrice': self.get_gas_price(),
                'nonce': self.w3.eth.get_transaction_count(self.w3.eth.accounts[0])
            })
            
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=os.getenv('PRIVATE_KEY'))
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            return self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def calculate_optimal_fee(self) -> float:
        """Calculate optimal protocol fee based on market conditions."""
        # Get market metrics
        metrics = self.get_treasury_metrics()
        
        # Base fee on TVL and competition
        base_fee = 0.002  # 0.2% base fee
        
        # Adjust based on TVL
        tvl_factor = min(1.5, max(0.5, metrics['total_reserves'] / 1000000))  # Scale based on $1M TVL
        
        # Adjust based on debt ratio
        debt_factor = min(1.2, max(0.8, 1 - metrics['debt_ratio']))
        
        # Calculate optimal fee
        optimal_fee = base_fee * tvl_factor * debt_factor
        
        # Ensure fee is within reasonable bounds
        return min(0.005, max(0.001, optimal_fee))  # Between 0.1% and 0.5%
    
    def manage_treasury(self):
        """
        Main function to manage treasury operations.
        """
        # Get current metrics
        metrics = self.get_treasury_metrics()
        
        # Prepare features for prediction
        features = self.prepare_features(metrics)
        
        # Predict if action is needed
        action_probability = self.predict_treasury_action(features)
        
        actions_taken = []
        
        if action_probability > 0.5:  # If probability > 50%
            # Optimize reserves
            reserve_optimization = self.optimize_reserves()
            
            if reserve_optimization:
                actions_taken.append('adjusted_reserves')
            
            # Optimize fee collection
            self.optimize_fee_collection()
            actions_taken.append('optimized_fees')
        
        return {
            'metrics': metrics,
            'action_probability': action_probability,
            'actions_taken': actions_taken
        } 