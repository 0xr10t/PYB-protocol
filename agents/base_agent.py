from web3 import Web3 
from typing import Dict, Any, List, Tuple
import json
import os
from dotenv import load_dotenv

class BaseAgent:
    def __init__(self, endpoint_uri, contract_addresses, is_testing=False):
        self.w3 = Web3(Web3.HTTPProvider(endpoint_uri))
        self.contract_addresses = contract_addresses
        self.is_testing = is_testing
        self.mock_data = None
        self.supported_tokens = [
            '0x1234567890123456789012345678901234567890',  # ETH
            '0x2468135790246813579024681357902468135790',  # WBTC
            '0x3579246813579024681357902468135790246813'   # USDC
        ]
        
        if not is_testing:
            self._load_contracts()
            
    def _load_contracts(self):
        """Load contract instances"""
        self.contracts = {}
        abi_dir = os.path.join(os.path.dirname(__file__), '..', 'test', 'abi')
        
        # Load StrategyManager ABI
        strategy_manager_path = os.path.join(abi_dir, 'StrategyManager.json')
        with open(strategy_manager_path, 'r') as f:
            strategy_manager_abi = json.load(f)
        self.contracts['strategy_manager'] = self.w3.eth.contract(
            address=self.contract_addresses['strategy_manager'],
            abi=strategy_manager_abi
        )
        
        # Load Treasury ABI
        treasury_path = os.path.join(abi_dir, 'Treasury.json')
        with open(treasury_path, 'r') as f:
            treasury_abi = json.load(f)
        self.contracts['treasury'] = self.w3.eth.contract(
            address=self.contract_addresses['treasury'],
            abi=treasury_abi
        )
        
        # Load BondFactory ABI
        bond_factory_path = os.path.join(abi_dir, 'BondFactory.json')
        with open(bond_factory_path, 'r') as f:
            bond_factory_abi = json.load(f)
        self.contracts['bond_factory'] = self.w3.eth.contract(
            address=self.contract_addresses['bond_factory'],
            abi=bond_factory_abi
        )
    
    def get_contract(self, name: str):
        """Get a contract instance by name."""
        return self.contracts.get(name)
    
    def get_block_number(self) -> int:
        """Get the current block number."""
        return self.w3.eth.block_number
    
    def get_token_balance(self, token_address: str, account_address: str) -> int:
        """Get token balance for an account."""
        token_contract = self.get_contract('ERC20')
        return token_contract.functions.balanceOf(account_address).call()
    
    def estimate_gas(self, transaction: Dict[str, Any]) -> int:
        """Estimate gas for a transaction."""
        return self.w3.eth.estimate_gas(transaction)
    
    def get_gas_price(self) -> int:
        """Get current gas price."""
        return self.w3.eth.gas_price
    
    def format_amount(self, amount):
        """Format amount from wei to ether"""
        return float(Web3.from_wei(amount, 'ether'))
        
    def parse_amount(self, amount):
        """Parse amount from ether to wei"""
        return Web3.to_wei(amount, 'ether')
    
    def load_abi(self, contract_name):
        """Load ABI for a specific contract"""
        if self.is_testing:
            return []
            
        abi_dir = os.path.join(os.path.dirname(__file__), '..', 'test', 'abi')
        abi_path = os.path.join(abi_dir, f'{contract_name}.json')
        
        with open(abi_path, 'r') as f:
            return json.load(f) 