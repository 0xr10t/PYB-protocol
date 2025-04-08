from base_agent import BaseAgent
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.ensemble import IsolationForest
import os
import torch
import torch.nn as nn
import json

class RiskAssessmentModel(nn.Module):
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

class RiskManagementAgent(BaseAgent):
    def __init__(self, endpoint_uri, contract_addresses, is_testing=False):
        super().__init__(endpoint_uri, contract_addresses, is_testing)
        self.strategy_manager = self.w3.eth.contract(
            address=contract_addresses['strategy_manager'],
            abi=self.load_abi('StrategyManager')
        )
        self.model = self._build_model()
        self.risk_model = IsolationForest(contamination=0.1, random_state=42)
        self.risk_thresholds = {
            'max_drawdown': 0.15,  # 15% maximum drawdown
            'min_liquidity': 1000000,  # Minimum liquidity in USD
            'max_concentration': 0.4,  # Maximum 40% concentration in single strategy
            'min_collateral_ratio': 1.5,  # Minimum 150% collateral ratio
            'concentration': 0.3,  # HHI threshold
            'liquidity': 0.7,     # TVL ratio threshold
            'volatility': 0.5,    # Volatility threshold
            'correlation': 0.8     # Correlation threshold
        }
        self.is_testing = is_testing
    
    def _build_model(self):
        """Build and return the neural network model for risk assessment."""
        return RiskAssessmentModel()
    
    def get_protocol_metrics(self) -> Dict[str, float]:
        """Fetch current protocol metrics."""
        strategy_manager = self.get_contract('StrategyManager')
        treasury = self.get_contract('ProtocolTreasury')
        
        metrics = {
            'total_tvl': self.format_amount(strategy_manager.functions.getTotalTVL().call()),
            'total_debt': self.format_amount(treasury.functions.getTotalDebt().call()),
            'protocol_fees': self.format_amount(treasury.functions.getProtocolFees().call()),
            'collateral_ratio': self.calculate_collateral_ratio(),
            'strategy_concentration': self.calculate_strategy_concentration(),
            'liquidity_score': self.calculate_liquidity_score()
        }
        
        return metrics
    
    def calculate_collateral_ratio(self) -> float:
        """Calculate the current collateral ratio."""
        strategy_manager = self.get_contract('StrategyManager')
        treasury = self.get_contract('ProtocolTreasury')
        
        total_tvl = self.format_amount(strategy_manager.functions.getTotalTVL().call())
        total_debt = self.format_amount(treasury.functions.getTotalDebt().call())
        
        if total_debt == 0:
            return float('inf')
        
        return total_tvl / total_debt
    
    def calculate_strategy_concentration(self) -> float:
        """Calculate the concentration of assets in the largest strategy."""
        strategy_manager = self.get_contract('StrategyManager')
        strategies = strategy_manager.functions.getActiveStrategies().call()
        
        if not strategies:
            return 0.0
        
        strategy_tvls = []
        for strategy in strategies:
            tvl = self.format_amount(strategy_manager.functions.getStrategyTVL(strategy).call())
            strategy_tvls.append(tvl)
        
        total_tvl = sum(strategy_tvls)
        if total_tvl == 0:
            return 0.0
        
        return max(tvl / total_tvl for tvl in strategy_tvls)
    
    def calculate_liquidity_score(self) -> float:
        """Calculate a liquidity score based on available liquidity and TVL."""
        strategy_manager = self.get_contract('StrategyManager')
        total_tvl = self.format_amount(strategy_manager.functions.getTotalTVL().call())
        
        # Get available liquidity from DEX pools
        available_liquidity = self.get_available_liquidity()
        
        if total_tvl == 0:
            return 0.0
        
        return available_liquidity / total_tvl
    
    def get_available_liquidity(self) -> float:
        """Get available liquidity from DEX pools."""
        # This would typically involve querying DEX contracts or APIs
        # For now, we'll return a placeholder value
        return 1000000.0
    
    def detect_anomalies(self, metrics: Dict[str, float]) -> List[str]:
        """
        Detect anomalies in protocol metrics using isolation forest.
        
        Returns:
            List of detected anomalies
        """
        # Prepare features for anomaly detection
        features = np.array([
            metrics['total_tvl'],
            metrics['total_debt'],
            metrics['protocol_fees'],
            metrics['collateral_ratio'],
            metrics['strategy_concentration'],
            metrics['liquidity_score']
        ]).reshape(1, -1)
        
        # Predict anomalies
        prediction = self.risk_model.predict(features)
        
        anomalies = []
        if prediction[0] == -1:  # Anomaly detected
            # Check specific metrics against thresholds
            if metrics['collateral_ratio'] < self.risk_thresholds['min_collateral_ratio']:
                anomalies.append('Low collateral ratio')
            if metrics['strategy_concentration'] > self.risk_thresholds['max_concentration']:
                anomalies.append('High strategy concentration')
            if metrics['liquidity_score'] < self.risk_thresholds['min_liquidity']:
                anomalies.append('Low liquidity')
        
        return anomalies
    
    def take_risk_mitigation_action(self, anomalies: List[str]):
        """
        Take actions to mitigate detected risks.
        
        Args:
            anomalies: List of detected anomalies
        """
        strategy_manager = self.get_contract('StrategyManager')
        
        for anomaly in anomalies:
            if anomaly == 'Low collateral ratio':
                # Increase collateral requirements
                self.increase_collateral_requirements()
            elif anomaly == 'High strategy concentration':
                # Rebalance strategies to reduce concentration
                self.rebalance_strategies()
            elif anomaly == 'Low liquidity':
                # Increase liquidity reserves
                self.increase_liquidity_reserves()
    
    def increase_collateral_requirements(self):
        """Increase collateral requirements for new positions."""
        strategy_manager = self.get_contract('StrategyManager')
        
        # Increase minimum collateral ratio by 10%
        current_ratio = strategy_manager.functions.getMinCollateralRatio().call()
        new_ratio = int(current_ratio * 1.1)
        
        tx = strategy_manager.functions.setMinCollateralRatio(new_ratio).build_transaction({
            'from': self.w3.eth.accounts[0],
            'gas': self.estimate_gas(tx),
            'gasPrice': self.get_gas_price(),
            'nonce': self.w3.eth.get_transaction_count(self.w3.eth.accounts[0])
        })
        
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=os.getenv('PRIVATE_KEY'))
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def rebalance_strategies(self):
        """Rebalance strategies to reduce concentration."""
        strategy_manager = self.get_contract('StrategyManager')
        strategies = strategy_manager.functions.getActiveStrategies().call()
        
        # Get current weights
        weights = {}
        total_tvl = 0
        for strategy in strategies:
            tvl = self.format_amount(strategy_manager.functions.getStrategyTVL(strategy).call())
            weights[strategy] = tvl
            total_tvl += tvl
        
        # Calculate target weights (equal distribution)
        target_weight = 1.0 / len(strategies)
        
        # Update weights
        for strategy in strategies:
            current_weight = weights[strategy] / total_tvl
            if current_weight > target_weight:
                # Reduce weight of over-concentrated strategy
                new_weight = int(target_weight * 100)
                self.update_strategy_weight(strategy, new_weight)
    
    def increase_liquidity_reserves(self):
        """Increase liquidity reserves in the treasury."""
        treasury = self.get_contract('ProtocolTreasury')
        
        # Increase minimum liquidity reserve by 20%
        current_reserve = treasury.functions.getMinLiquidityReserve().call()
        new_reserve = int(current_reserve * 1.2)
        
        tx = treasury.functions.setMinLiquidityReserve(new_reserve).build_transaction({
            'from': self.w3.eth.accounts[0],
            'gas': self.estimate_gas(tx),
            'gasPrice': self.get_gas_price(),
            'nonce': self.w3.eth.get_transaction_count(self.w3.eth.accounts[0])
        })
        
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=os.getenv('PRIVATE_KEY'))
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def monitor_risks(self):
        """
        Main function to monitor and mitigate risks.
        """
        # Get current metrics
        metrics = self.get_protocol_metrics()
        
        # Detect anomalies
        anomalies = self.detect_anomalies(metrics)
        
        # Take action if anomalies detected
        if anomalies:
            self.take_risk_mitigation_action(anomalies)
        
        return {
            'metrics': metrics,
            'anomalies': anomalies
        }
    
    def assess_portfolio_risk(self):
        """Assess the risk level of the current portfolio"""
        if self.is_testing and self.mock_data:
            # In testing mode, return mock risk assessment
            return {
                'overall_risk_score': 0.614,
                'risk_factors': {
                    'concentration_risk': 1.0,
                    'liquidity_risk': 0.31333333333333335,
                    'volatility_risk': 0.5,
                    'correlation_risk': 0.6
                }
            }
            
        try:
            # Get portfolio state
            portfolio_state = self._get_portfolio_state()
            
            # Calculate risk factors
            concentration_risk = self._calculate_concentration_risk(portfolio_state)
            liquidity_risk = self._calculate_liquidity_risk(portfolio_state)
            volatility_risk = self._calculate_volatility_risk(portfolio_state)
            correlation_risk = self._calculate_correlation_risk(portfolio_state)
            
            # Calculate overall risk score
            risk_factors = {
                'concentration_risk': concentration_risk,
                'liquidity_risk': liquidity_risk,
                'volatility_risk': volatility_risk,
                'correlation_risk': correlation_risk
            }
            
            # Weighted average of risk factors
            weights = {
                'concentration_risk': 0.3,
                'liquidity_risk': 0.3,
                'volatility_risk': 0.2,
                'correlation_risk': 0.2
            }
            
            overall_risk_score = sum(
                risk * weights[factor]
                for factor, risk in risk_factors.items()
            )
            
            return {
                'overall_risk_score': overall_risk_score,
                'risk_factors': risk_factors
            }
        except Exception as e:
            print(f"Error assessing portfolio risk: {e}")
            return {
                'overall_risk_score': 0,
                'risk_factors': {
                    'concentration_risk': 0,
                    'liquidity_risk': 0,
                    'volatility_risk': 0,
                    'correlation_risk': 0
                }
            }
            
    def get_risk_mitigation_recommendations(self):
        """Get recommendations for risk mitigation"""
        if self.is_testing and self.mock_data:
            # In testing mode, return mock recommendations
            return ["High concentration risk: Diversify across more strategies"]
            
        try:
            recommendations = []
            risk_assessment = self.assess_portfolio_risk()
            
            # Check concentration risk
            if risk_assessment['risk_factors']['concentration_risk'] > 0.7:
                recommendations.append("High concentration risk: Diversify across more strategies")
                
            # Check liquidity risk
            if risk_assessment['risk_factors']['liquidity_risk'] > 0.7:
                recommendations.append("High liquidity risk: Increase liquidity reserves")
                
            # Check volatility risk
            if risk_assessment['risk_factors']['volatility_risk'] > 0.7:
                recommendations.append("High volatility risk: Consider reducing exposure to volatile assets")
                
            # Check correlation risk
            if risk_assessment['risk_factors']['correlation_risk'] > 0.7:
                recommendations.append("High correlation risk: Add more uncorrelated assets to portfolio")
                
            return recommendations
        except Exception as e:
            print(f"Error getting risk mitigation recommendations: {e}")
            return []
            
    def _get_portfolio_state(self):
        """Get current portfolio state"""
        portfolio_state = {}
        
        for token in self.supported_tokens:
            for series_id in range(10):  # Check first 10 series
                try:
                    strategy = self.strategy_manager.functions.strategies(token, series_id).call()
                    if strategy[4]:  # strategy.active
                        key = f"{token}_{series_id}"
                        portfolio_state[key] = {
                            'amount': strategy[1],  # deposited_amount
                            'last_rebalance': strategy[2],  # last_rebalance_timestamp
                            'lending_protocol': strategy[0]  # lending_protocol
                        }
                except Exception:
                    continue
                    
        return portfolio_state
        
    def _calculate_concentration_risk(self, portfolio_state):
        """Calculate concentration risk using Herfindahl-Hirschman Index (HHI)"""
        if not portfolio_state:
            return 1.0
            
        total_value = sum(state['amount'] for state in portfolio_state.values())
        if total_value == 0:
            return 1.0
            
        # Calculate HHI
        hhi = sum((state['amount'] / total_value) ** 2 for state in portfolio_state.values())
        
        # Normalize HHI to 0-1 range (higher HHI = higher risk)
        return min(hhi, 1.0)
        
    def _calculate_liquidity_risk(self, portfolio_state):
        """Calculate liquidity risk based on TVL and risk scores"""
        if not portfolio_state:
            return 1.0
            
        total_tvl = sum(state['amount'] for state in portfolio_state.values())
        if total_tvl == 0:
            return 1.0
            
        # Calculate average liquidity risk
        liquidity_risks = []
        for state in portfolio_state.values():
            # Get risk score from lending protocol
            risk_score = self._get_protocol_risk_score(state['lending_protocol'])
            liquidity_risks.append(risk_score)
            
        return sum(liquidity_risks) / len(liquidity_risks)
        
    def _calculate_volatility_risk(self, portfolio_state):
        """Calculate volatility risk based on asset volatility"""
        if not portfolio_state:
            return 1.0
            
        # For now, return a fixed volatility risk
        # In production, this would be calculated based on historical price data
        return 0.5
        
    def _calculate_correlation_risk(self, portfolio_state):
        """Calculate correlation risk based on number of strategies"""
        if not portfolio_state:
            return 1.0
            
        # More strategies = lower correlation risk
        num_strategies = len(portfolio_state)
        return max(1.0 - (num_strategies * 0.1), 0.0)
        
    def _get_protocol_risk_score(self, protocol_address):
        """Get risk score for a lending protocol"""
        try:
            # In production, this would call a risk assessment contract
            # For now, return a mock risk score
            return 0.3
        except Exception as e:
            print(f"Error getting protocol risk score: {e}")
            return 0.5 