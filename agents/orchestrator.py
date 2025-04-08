from typing import Dict, List
import os
from dotenv import load_dotenv
from yield_strategy_agent import YieldStrategyAgent
from risk_management_agent import RiskManagementAgent
from rebalancing_agent import RebalancingAgent
from price_oracle_agent import PriceOracleAgent
from treasury_management_agent import TreasuryManagementAgent

class Orchestrator:
    def __init__(self, web3_provider: str, contract_addresses: Dict[str, str]):
        """
        Initialize the orchestrator with all agents.
        
        Args:
            web3_provider: Web3 provider URL
            contract_addresses: Dictionary of contract addresses
        """
        load_dotenv()
        
        self.web3_provider = web3_provider
        self.contract_addresses = contract_addresses
        
        # Initialize agents
        self.yield_strategy_agent = YieldStrategyAgent(web3_provider, contract_addresses)
        self.risk_management_agent = RiskManagementAgent(web3_provider, contract_addresses)
        self.rebalancing_agent = RebalancingAgent(web3_provider, contract_addresses)
        self.price_oracle_agent = PriceOracleAgent(web3_provider, contract_addresses)
        self.treasury_management_agent = TreasuryManagementAgent(web3_provider, contract_addresses)
        
        # Track agent states
        self.agent_states = {
            'yield_strategy': {},
            'risk_management': {},
            'rebalancing': {},
            'price_oracle': {},
            'treasury_management': {}
        }
    
    def run_yield_optimization(self) -> Dict:
        """
        Run yield strategy optimization.
        
        Returns:
            Dictionary of optimization results
        """
        # Get market data and optimize portfolio
        market_data = self.yield_strategy_agent.get_market_data()
        optimization_result = self.yield_strategy_agent.optimize_portfolio()
        
        # Update agent state
        self.agent_states['yield_strategy'] = {
            'market_data': market_data.to_dict(),
            'optimization_result': optimization_result
        }
        
        return self.agent_states['yield_strategy']
    
    def run_risk_management(self) -> Dict:
        """
        Run risk management operations.
        
        Returns:
            Dictionary of risk management results
        """
        # Monitor and mitigate risks
        risk_results = self.risk_management_agent.monitor_risks()
        
        # Update agent state
        self.agent_states['risk_management'] = risk_results
        
        return risk_results
    
    def run_rebalancing(self) -> Dict:
        """
        Run portfolio rebalancing operations.
        
        Returns:
            Dictionary of rebalancing results
        """
        # Monitor and execute rebalancing
        rebalancing_results = self.rebalancing_agent.monitor_and_rebalance()
        
        # Update agent state
        self.agent_states['rebalancing'] = rebalancing_results
        
        return rebalancing_results
    
    def run_price_monitoring(self, tokens: List[str]) -> Dict:
        """
        Run price monitoring and prediction.
        
        Args:
            tokens: List of tokens to monitor
            
        Returns:
            Dictionary of price monitoring results
        """
        # Monitor prices and predictions
        price_results = self.price_oracle_agent.monitor_prices(tokens)
        
        # Update agent state
        self.agent_states['price_oracle'] = price_results
        
        return price_results
    
    def run_treasury_management(self) -> Dict:
        """
        Run treasury management operations.
        
        Returns:
            Dictionary of treasury management results
        """
        # Manage treasury operations
        treasury_results = self.treasury_management_agent.manage_treasury()
        
        # Update agent state
        self.agent_states['treasury_management'] = treasury_results
        
        return treasury_results
    
    def run_full_cycle(self) -> Dict:
        """
        Run a full cycle of all agent operations.
        
        Returns:
            Dictionary of results from all agents
        """
        results = {
            'yield_strategy': self.run_yield_optimization(),
            'risk_management': self.run_risk_management(),
            'rebalancing': self.run_rebalancing(),
            'price_oracle': self.run_price_monitoring(['ETH', 'WBTC', 'USDC']),
            'treasury_management': self.run_treasury_management()
        }
        
        return results
    
    def get_agent_states(self) -> Dict:
        """
        Get the current state of all agents.
        
        Returns:
            Dictionary of agent states
        """
        return self.agent_states
    
    def get_agent_state(self, agent_name: str) -> Dict:
        """
        Get the current state of a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dictionary of agent state
        """
        return self.agent_states.get(agent_name, {})
    
    def reset_agent_states(self):
        """Reset all agent states."""
        self.agent_states = {
            'yield_strategy': {},
            'risk_management': {},
            'rebalancing': {},
            'price_oracle': {},
            'treasury_management': {}
        } 