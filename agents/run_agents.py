import os
import time
from dotenv import load_dotenv
from orchestrator import Orchestrator

def main():
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment variables
    web3_provider = os.getenv('WEB3_PROVIDER')
    if not web3_provider:
        raise ValueError("WEB3_PROVIDER environment variable not set")
    
    # Get contract addresses from environment variables
    contract_addresses = {
        'StrategyManager': os.getenv('STRATEGY_MANAGER_ADDRESS'),
        'ProtocolTreasury': os.getenv('PROTOCOL_TREASURY_ADDRESS'),
        'YieldDistribution': os.getenv('YIELD_DISTRIBUTION_ADDRESS'),
        'BondFactory': os.getenv('BOND_FACTORY_ADDRESS')
    }
    
    # Validate contract addresses
    for contract_name, address in contract_addresses.items():
        if not address:
            raise ValueError(f"{contract_name}_ADDRESS environment variable not set")
    
    # Initialize orchestrator
    orchestrator = Orchestrator(web3_provider, contract_addresses)
    
    # Run agents continuously
    while True:
        try:
            print("\nRunning full agent cycle...")
            
            # Run full cycle
            results = orchestrator.run_full_cycle()
            
            # Print results
            print("\nResults:")
            for agent_name, agent_results in results.items():
                print(f"\n{agent_name.upper()}:")
                if isinstance(agent_results, dict):
                    for key, value in agent_results.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {agent_results}")
            
            # Wait for next cycle
            print("\nWaiting for next cycle...")
            time.sleep(300)  # 5 minutes between cycles
            
        except Exception as e:
            print(f"Error in agent cycle: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    main() 