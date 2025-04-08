from base_agent import BaseAgent
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import os
import requests
from datetime import datetime, timedelta

class PricePredictionModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=128):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size//2, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(hidden_size//2, hidden_size//4, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size//4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.fc1(x[:, -1, :])
        x = self.relu(x)
        x = self.fc2(x)
        return x

class PriceOracleAgent(BaseAgent):
    def __init__(self, web3_provider: str, contract_addresses: Dict[str, str], is_testing: bool = False, mock_data: Dict = None):
        super().__init__(web3_provider, contract_addresses, is_testing)
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.is_testing = is_testing
        self.mock_data = mock_data
        self.price_feeds = {
            'ETH': '0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419',  # ETH/USD
            'WBTC': '0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c',  # BTC/USD
            'USDC': '0x8fFfFfd4AfB6115b954Bd326cbe7B4BA576818f6'   # USDC/USD
        }
        
    def _build_model(self) -> nn.Module:
        """Build and return the neural network model for price prediction."""
        model = PricePredictionModel()
        return model
    
    def get_price_data(self, token: str, days: int = 60) -> pd.DataFrame:
        """
        Fetch historical price data for a token.
        
        Args:
            token: Token symbol
            days: Number of days of historical data to fetch
        """
        # Get price feed address
        price_feed = self.price_feeds.get(token)
        if not price_feed:
            raise ValueError(f"No price feed found for token {token}")
        
        # Get price feed contract
        price_feed_contract = self.get_contract('ChainlinkAggregatorV3')
        price_feed_contract.address = price_feed
        
        # Get historical price data
        price_data = []
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        current_time = start_time
        while current_time < end_time:
            try:
                # Get price at timestamp
                price = price_feed_contract.functions.getHistoricalPrice(
                    int(current_time.timestamp())
                ).call()
                
                price_data.append({
                    'timestamp': current_time.timestamp(),
                    'price': self.format_amount(price)
                })
                
                current_time += timedelta(hours=1)
            except Exception as e:
                print(f"Error fetching price data: {e}")
                break
        
        return pd.DataFrame(price_data)
    
    def get_market_data(self, token: str) -> Dict[str, float]:
        """
        Fetch current market data for a token.
        
        Args:
            token: Token symbol
        """
        # Get price feed address
        price_feed = self.price_feeds.get(token)
        if not price_feed:
            raise ValueError(f"No price feed found for token {token}")
        
        # Get price feed contract
        price_feed_contract = self.get_contract('ChainlinkAggregatorV3')
        price_feed_contract.address = price_feed
        
        # Get current price and other market data
        price = price_feed_contract.functions.latestAnswer().call()
        decimals = price_feed_contract.functions.decimals().call()
        
        # Get 24h price change
        day_ago_price = price_feed_contract.functions.getHistoricalPrice(
            int((datetime.now() - timedelta(days=1)).timestamp())
        ).call()
        
        price_change_24h = (price - day_ago_price) / day_ago_price if day_ago_price > 0 else 0
        
        return {
            'price': self.format_amount(price),
            'price_change_24h': price_change_24h,
            'decimals': decimals
        }
    
    def prepare_features(self, price_data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for the price prediction model.
        
        Args:
            price_data: DataFrame containing historical price data
        """
        # Calculate technical indicators
        price_data['returns'] = price_data['price'].pct_change()
        price_data['volatility'] = price_data['returns'].rolling(window=24).std()
        price_data['ma_7'] = price_data['price'].rolling(window=7*24).mean()
        price_data['ma_30'] = price_data['price'].rolling(window=30*24).mean()
        price_data['rsi'] = self.calculate_rsi(price_data['price'])
        
        # Select features for model
        features = price_data[['price', 'returns', 'volatility', 'ma_7', 'ma_30', 'rsi']].values
        
        # Reshape for LSTM input (samples, timesteps, features)
        features = features.reshape(-1, 60, 6)
        
        return features
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def predict_price(self, features: np.ndarray) -> float:
        """
        Predict future price.
        
        Args:
            features: Prepared feature array
            
        Returns:
            Predicted price
        """
        # Scale features
        features_scaled = self.scaler.fit_transform(features.reshape(-1, features.shape[-1]))
        features_scaled = features_scaled.reshape(features.shape)
        
        # Convert to PyTorch tensor
        features_tensor = torch.FloatTensor(features_scaled)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Predict price
        with torch.no_grad():
            prediction = self.model(features_tensor)
        
        return float(prediction[0][0])
    
    def get_price_prediction(self, token: str) -> Dict[str, float]:
        """
        Get price prediction for a token.
        
        Args:
            token: Token symbol
            
        Returns:
            Dictionary containing current price and prediction
        """
        # Get historical price data
        price_data = self.get_price_data(token)
        
        # Get current market data
        market_data = self.get_market_data(token)
        
        # Prepare features
        features = self.prepare_features(price_data)
        
        # Predict price
        predicted_price = self.predict_price(features)
        
        return {
            'current_price': market_data['price'],
            'predicted_price': predicted_price,
            'price_change_24h': market_data['price_change_24h'],
            'prediction_change': (predicted_price - market_data['price']) / market_data['price']
        }
    
    def monitor_prices(self, tokens: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Monitor prices and predictions for multiple tokens.
        
        Args:
            tokens: List of token symbols to monitor
            
        Returns:
            Dictionary of price data and predictions for each token
        """
        results = {}
        
        for token in tokens:
            try:
                results[token] = self.get_price_prediction(token)
            except Exception as e:
                print(f"Error monitoring {token}: {e}")
                results[token] = {'error': str(e)}
        
        return results
    
    def should_rebalance(self, token: str, current_weight: float) -> Tuple[bool, float]:
        """
        Determine if a token's position should be rebalanced based on price predictions.
        
        Args:
            token: Token symbol
            current_weight: Current portfolio weight of the token
            
        Returns:
            Tuple of (should_rebalance, target_weight)
        """
        # Get price prediction
        prediction = self.get_price_prediction(token)
        
        # Calculate expected return
        expected_return = prediction['prediction_change']
        
        # Define rebalancing thresholds
        min_weight = 0.05  # Minimum 5% weight
        max_weight = 0.40  # Maximum 40% weight
        
        # Calculate target weight based on expected return
        if expected_return > 0.1:  # If expected return > 10%
            target_weight = min(max_weight, current_weight * (1 + expected_return))
        elif expected_return < -0.1:  # If expected return < -10%
            target_weight = max(min_weight, current_weight * (1 + expected_return))
        else:
            target_weight = current_weight
        
        # Check if rebalancing is needed
        should_rebalance = abs(target_weight - current_weight) > 0.05  # 5% deviation threshold
        
        return should_rebalance, target_weight
    
    def get_token_price(self, token: str) -> float:
        """
        Get the current price of a token.
        
        Args:
            token: Token symbol (e.g., 'ETH', 'WBTC')
        
        Returns:
            Current token price in USD
        """
        if self.is_testing and self.mock_data:
            return self.mock_data['prices'][token][0]
        
        # Real contract interaction code
        bond_factory = self.get_contract('BondFactory')
        try:
            price = bond_factory.functions.getBondPrice(token).call()
            return self.format_amount(price)
        except Exception as e:
            print(f"Error getting price for {token}: {e}")
            return 0.0 