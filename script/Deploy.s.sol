// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "../src/BondFactory.sol";
import "../src/YieldDistribution.sol";
import "../src/StrategyManager.sol";
import "../src/ProtocolTreasury.sol";

contract DeployScript is Script {
    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        vm.startBroadcast(deployerPrivateKey);

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
            tempBondFactory, // Temporary address that will be updated
            100 // 1% protocol fee
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
        treasury.addSupportedToken(address(0)); // ETH
        strategyManager.addSupportedToken(address(0)); // ETH

        vm.stopBroadcast();

        console.log("Deployment Addresses:");
        console.log("StrategyManager:", address(strategyManager));
        console.log("Treasury:", address(treasury));
        console.log("YieldDistribution:", address(yieldDistribution));
        console.log("BondFactory:", address(factory));
    }
}

// Temporary proxy contract to provide a non-zero address
contract TemporaryProxy {} 