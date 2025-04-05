// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../src/BondFactory.sol";
import "../src/BondToken.sol";
import "../src/YieldDistribution.sol";
import "../src/StrategyManager.sol";
import "../src/ProtocolTreasury.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

// Mock ERC20 token for testing
contract MockERC20 is ERC20 {
    constructor(string memory name, string memory symbol) ERC20(name, symbol) {
        _mint(msg.sender, 1000000 * 10**18);
    }
}

// Mock lending protocol for testing
contract MockLendingProtocol {
    mapping(address => uint256) public deposits;
    
    function deposit(address token, uint256 amount) external {
        deposits[token] += amount;
        IERC20(token).transferFrom(msg.sender, address(this), amount);
    }
    
    function withdraw(address token, uint256 amount) external {
        require(deposits[token] >= amount, "Insufficient deposits");
        deposits[token] -= amount;
        IERC20(token).transfer(msg.sender, amount);
    }
}

contract PYBProtocolTest is Test {
    BondFactory public factory;
    YieldDistribution public yieldDistribution;
    StrategyManager public strategyManager;
    ProtocolTreasury public treasury;
    
    MockERC20 public mockToken;
    MockLendingProtocol public mockLendingProtocol;
    
    address public owner;
    address public user1;
    address public user2;
    
    function setUp() public {
        owner = address(this);
        user1 = address(0x1);
        user2 = address(0x2);
        
        // Deploy mock token
        mockToken = new MockERC20("Mock Token", "MTK");
        
        // Deploy mock lending protocol
        mockLendingProtocol = new MockLendingProtocol();
        
        // Deploy protocol contracts in the correct order
        vm.startPrank(owner);
        
        strategyManager = new StrategyManager(1 days, 1 ether);
        treasury = new ProtocolTreasury(address(0), address(strategyManager));
        
        // Deploy YieldDistribution with temporary factory address
        yieldDistribution = new YieldDistribution(
            address(treasury),
            address(this), // Temporary factory address (will be updated)
            100 // 1% protocol fee
        );
        
        // Deploy BondFactory with all required dependencies
        factory = new BondFactory(
            address(treasury),
            address(yieldDistribution),
            address(strategyManager)
        );
        
        // Update contract references and ownership
        yieldDistribution.setBondFactory(address(factory));
        treasury.setYieldDistribution(address(yieldDistribution));
        strategyManager.setBondFactory(address(factory));
        
        // Set up initial configuration
        treasury.addSupportedToken(address(mockToken));
        strategyManager.addSupportedToken(address(mockToken));
        
        // Set up strategy
        strategyManager.setStrategy(address(mockToken), 0, address(mockLendingProtocol));
        
        vm.stopPrank();
        
        // Transfer tokens to users
        mockToken.transfer(user1, 1000 * 10**18);
        mockToken.transfer(user2, 1000 * 10**18);
    }
    
    function testCreateBondSeries() public {
        vm.startPrank(owner);
        uint256 seriesId = factory.createBondSeries(address(mockToken), 500); // 5% yield
        
        (
            ,
            address collateralToken,
            uint256 initialYield,
            uint256 totalDeposits,
            bool active
        ) = factory.getBondSeries(seriesId);
        
        assertEq(collateralToken, address(mockToken));
        assertEq(initialYield, 500);
        assertEq(totalDeposits, 0);
        assertEq(active, true);
        vm.stopPrank();
    }
    
    function testDepositCollateral() public {
        vm.startPrank(owner);
        uint256 seriesId = factory.createBondSeries(address(mockToken), 500); // 5% yield
        vm.stopPrank();
        
        vm.startPrank(user1);
        mockToken.approve(address(factory), 100 * 10**18);
        factory.depositCollateral(seriesId, 100 * 10**18);
        vm.stopPrank();
        
        (
            address bondToken,
            ,
            ,
            uint256 totalDeposits,
            bool active
        ) = factory.getBondSeries(seriesId);
        
        assertEq(totalDeposits, 100 * 10**18);
        
        // Check bond NFT
        BondToken bondTokenContract = BondToken(bondToken);
        assertEq(bondTokenContract.ownerOf(1), user1);
        
        (
            uint256 amount,
            uint256 yieldRate,
            ,
            uint256 unclaimedYield
        ) = bondTokenContract.getBondDetails(1);
        
        assertEq(amount, 100 * 10**18);
        assertEq(yieldRate, 500);
        assertEq(unclaimedYield, 0);
    }
    
    function testYieldDistribution() public {
        vm.startPrank(owner);
        uint256 seriesId = factory.createBondSeries(address(mockToken), 500); // 5% yield
        vm.stopPrank();
        
        vm.startPrank(user1);
        mockToken.approve(address(factory), 100 * 10**18);
        factory.depositCollateral(seriesId, 100 * 10**18);
        vm.stopPrank();
        
        (
            address bondToken,
            ,
            ,
            ,
            bool active
        ) = factory.getBondSeries(seriesId);
        
        // Simulate yield generation
        vm.startPrank(owner);
        uint256 yieldAmount = 10 * 10**18;
        mockToken.transfer(address(yieldDistribution), yieldAmount);
        
        // Set yield token
        yieldDistribution.setYieldToken(bondToken, address(mockToken));
        
        // Distribute yield
        yieldDistribution.distributeYield(bondToken, 1, yieldAmount);
        vm.stopPrank();
        
        // Advance time by 15 days
        vm.warp(block.timestamp + 15 days);
        
        // Check yield stream
        (
            uint256 totalYield,
            uint256 lastDistributionTimestamp,
            uint256 yieldPerSecond
        ) = yieldDistribution.getYieldStream(bondToken, 1);
        
        assertEq(totalYield, yieldAmount);
        
        // Claim yield
        vm.startPrank(user1);
        yieldDistribution.claimYield(bondToken, 1);
        vm.stopPrank();
        
        // Check user received yield (minus protocol fee)
        uint256 expectedYield = (yieldAmount * 15 days) / (30 days); // Half of the yield since we waited 15 days
        expectedYield = expectedYield - (expectedYield * 100) / 10000; // 1% fee
        assertEq(mockToken.balanceOf(user1), 1000 * 10**18 - 100 * 10**18 + expectedYield);
    }
    
    function testStrategyRebalancing() public {
        vm.startPrank(owner);
        uint256 seriesId = factory.createBondSeries(address(mockToken), 500); // 5% yield
        vm.stopPrank();
        
        vm.startPrank(user1);
        mockToken.approve(address(factory), 100 * 10**18);
        factory.depositCollateral(seriesId, 100 * 10**18);
        vm.stopPrank();
        
        // Check strategy deployment
        (
            address lendingProtocol,
            uint256 depositedAmount,
            ,
            bool active
        ) = strategyManager.strategies(address(mockToken), seriesId);
        
        assertEq(lendingProtocol, address(mockLendingProtocol));
        assertEq(depositedAmount, 100 * 10**18);
        assertEq(active, true);
        
        // Simulate rebalancing
        vm.warp(block.timestamp + 2 days);
        strategyManager.rebalanceStrategy(address(mockToken), seriesId);
        
        // Check strategy after rebalancing
        (
            ,
            depositedAmount,
            ,
            active
        ) = strategyManager.strategies(address(mockToken), seriesId);
        
        assertEq(depositedAmount, 100 * 10**18); // No change in this test
    }
} 