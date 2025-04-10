// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "../src/mocks/MockERC20.sol";


import "../src/BondFactory.sol";
import "../src/YieldDistribution.sol";
import "../src/StrategyManager.sol";
import "../src/ProtocolTreasury.sol";
import "../src/mocks/MockLendingProtocol.sol";

contract DeployScript is Script {
    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        vm.startBroadcast(deployerPrivateKey);

        MockERC20 token = new MockERC20("Mock Token", "MTK", msg.sender);
        token.mint(msg.sender, 100_000 ether);

        // Deploy StrategyManager
        StrategyManager strategyManager = new StrategyManager(
            1 days, // rebalanceInterval
            1 ether // minRebalanceAmount
        );

        // Deploy a temporary proxy contract that will be replaced with BondFactory
        address tempBondFactory = address(new TemporaryProxy());

        // Deploy ProtocolTreasury with placeholder addresses
        ProtocolTreasury treasury = new ProtocolTreasury(
            address(0), // Will be updated with yieldDistribution
            address(strategyManager)
        );

        // Deploy YieldDistribution with temporary BondFactory
        YieldDistribution yieldDistribution = new YieldDistribution(
            address(treasury),
            tempBondFactory,
            100 // 1% protocol fee
        );

        // Deploy MockLendingProtocol with deployed token address
        MockLendingProtocol mockLending = new MockLendingProtocol(
            address(token),
            msg.sender
        );

        // Deploy BondFactory
        BondFactory factory = new BondFactory(
            address(treasury),
            address(yieldDistribution),
            address(strategyManager)
        );

        // Update contract references
        yieldDistribution.setBondFactory(address(factory));
        treasury.setYieldDistribution(address(yieldDistribution));

        // Set up initial configuration
        treasury.addSupportedToken(address(token));
        strategyManager.addSupportedToken(address(token));
        strategyManager.setStrategy(address(token), 0, address(mockLending));

        vm.stopBroadcast();

        console.log("Deployment Addresses:");
        console.log("Token:", address(token));
        console.log("StrategyManager:", address(strategyManager));
        console.log("Treasury:", address(treasury));
        console.log("YieldDistribution:", address(yieldDistribution));
        console.log("BondFactory:", address(factory));
        console.log("MockLendingProtocol:", address(mockLending));
    }
}

// Temporary proxy contract to provide a non-zero address
contract TemporaryProxy {}
