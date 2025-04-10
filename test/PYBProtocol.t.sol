// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "forge-std/console.sol";
import "../src/BondFactory.sol";
import "../src/BondToken.sol";
import "../src/YieldDistribution.sol";
import "../src/StrategyManager.sol";
import "../src/ProtocolTreasury.sol";
import "../src/mocks/MockLendingProtocol.sol";
import "../src/mocks/MockERC20.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

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
        mockToken = new MockERC20("Mock Token", "MTK", msg.sender);
        // Make PYBProtocolTest the owner so it can mint
        vm.prank(address(0x1804c8AB1F12E6bbf3894d4083f33e07309d1f38)); // Default deployer
        mockToken.transferOwnership(address(this));

        vm.startPrank(address(this)); // or use the exact test contract address
        mockToken.mint(address(this), 1e24); // Mint enough tokens to self
        vm.stopPrank();

        // Deploy mock lending protocol (now from imported contract)
        mockLendingProtocol = new MockLendingProtocol(
            address(mockToken),
            owner
        );

        // Deploy protocol contracts
        vm.startPrank(owner);

        strategyManager = new StrategyManager(1 days, 1 ether);
        treasury = new ProtocolTreasury(address(0), address(strategyManager));

        yieldDistribution = new YieldDistribution(
            address(treasury),
            address(this),
            100
        );

        factory = new BondFactory(
            address(treasury),
            address(yieldDistribution),
            address(strategyManager)
        );

        // Wire up dependencies
        yieldDistribution.setBondFactory(address(factory));
        treasury.setYieldDistribution(address(yieldDistribution));
        strategyManager.setBondFactory(address(factory));

        treasury.addSupportedToken(address(mockToken));
        strategyManager.addSupportedToken(address(mockToken));

        strategyManager.setStrategy(
            address(mockToken),
            0,
            address(mockLendingProtocol)
        );

        vm.stopPrank();

        // Fund users
        mockToken.transfer(user1, 1000 * 10 ** 18);
        mockToken.transfer(user2, 1000 * 10 ** 18);
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
        mockToken.approve(address(factory), 100 * 10 ** 18);
        factory.depositCollateral(seriesId, 100 * 10 ** 18);
        vm.stopPrank();

        (address bondToken, , , uint256 totalDeposits, bool active) = factory
            .getBondSeries(seriesId);

        assertEq(totalDeposits, 100 * 10 ** 18);

        // Check bond NFT
        BondToken bondTokenContract = BondToken(bondToken);
        assertEq(bondTokenContract.ownerOf(1), user1);

        (
            uint256 amount,
            uint256 yieldRate,
            ,
            uint256 unclaimedYield
        ) = bondTokenContract.getBondDetails(1);

        assertEq(amount, 100 * 10 ** 18);
        assertEq(yieldRate, 500);
        assertEq(unclaimedYield, 0);
    }

    function testYieldDistribution() public {
        vm.startPrank(owner);
        uint256 seriesId = factory.createBondSeries(address(mockToken), 500); // 5% yield
        vm.stopPrank();

        vm.startPrank(user1);
        mockToken.approve(address(factory), 100 * 10 ** 18);
        factory.depositCollateral(seriesId, 100 * 10 ** 18);
        vm.stopPrank();

        (address bondToken, , , , bool active) = factory.getBondSeries(
            seriesId
        );

        // Simulate yield generation
        vm.startPrank(owner);
        uint256 yieldAmount = 10 * 10 ** 18;
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
        assertEq(
            mockToken.balanceOf(user1),
            1000 * 10 ** 18 - 100 * 10 ** 18 + expectedYield
        );
    }

    function testStrategyRebalancing() public {
        // --- Setup ---
        vm.startPrank(owner);
        strategyManager.addSupportedToken(address(mockToken));
        strategyManager.setStrategy(
            address(mockToken),
            0,
            address(mockLendingProtocol)
        );
        strategyManager.setBondFactory(address(factory));
        vm.stopPrank();

        // --- Create bond and deposit collateral ---
        vm.startPrank(owner);
        uint256 seriesId = factory.createBondSeries(address(mockToken), 500); // 5% yield
        vm.stopPrank();

        vm.startPrank(user1);
        mockToken.mint(user1, 200 ether);
        mockToken.approve(address(factory), 100 ether);
        factory.depositCollateral(seriesId, 100 ether);
        vm.stopPrank();

        // Add mock liquidity to simulate yield availability
        mockToken.mint(address(mockLendingProtocol), 2 ether);

        // --- Warp time to accrue interest ---
        vm.warp(block.timestamp + 2 days); // simulate time passing

        // --- Capture pre-rebalance balance of owner ---
        uint256 ownerBalanceBefore = mockToken.balanceOf(owner);

        // --- Rebalance strategy ---
        vm.startPrank(owner);
        strategyManager.rebalanceStrategy(address(mockToken), seriesId);
        vm.stopPrank();

        // --- Post-rebalance validations ---
        (
            address strategyAddress,
            uint256 newDepositedAmount,
            ,
            bool active
        ) = strategyManager.strategies(address(mockToken), seriesId);

        assertEq(
            strategyAddress,
            address(mockLendingProtocol),
            "Strategy address mismatch"
        );
        assertEq(active, true, "Strategy should still be active");
        assertGt(newDepositedAmount, 100 ether, "No yield accrued");

        uint256 ownerBalanceAfter = mockToken.balanceOf(owner);
        assertGt(
            ownerBalanceAfter,
            ownerBalanceBefore,
            "Owner should have received yield from rebalance"
        );
    }
}
