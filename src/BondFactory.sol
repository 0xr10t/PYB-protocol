// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "./BondToken.sol";
import "./YieldDistribution.sol";
import "./StrategyManager.sol";

contract BondFactory is Ownable, ReentrancyGuard {
    // Events
    event BondSeriesCreated(
        address indexed bondToken,
        address indexed collateralToken,
        uint256 seriesId,
        uint256 initialYield
    );

    // State variables
    struct BondSeries {
        address bondToken;
        address collateralToken;
        uint256 initialYield;
        uint256 totalDeposits;
        bool active;
    }

    mapping(uint256 => BondSeries) public bondSeries;
    uint256 public seriesCounter;
    
    YieldDistribution public yieldDistribution;
    StrategyManager public strategyManager;
    address public treasury;

    // Constructor
    constructor(
        address _treasury,
        address _yieldDistribution,
        address _strategyManager
    ) Ownable(msg.sender) {
        require(_treasury != address(0), "Invalid treasury address");
        require(_yieldDistribution != address(0), "Invalid yield distribution address");
        require(_strategyManager != address(0), "Invalid strategy manager address");
        treasury = _treasury;
        yieldDistribution = YieldDistribution(_yieldDistribution);
        strategyManager = StrategyManager(_strategyManager);
    }

    // Functions
    function createBondSeries(
        address _collateralToken,
        uint256 _initialYield
    ) external onlyOwner returns (uint256) {
        require(_collateralToken != address(0), "Invalid collateral token");
        
        uint256 seriesId = seriesCounter++;
        
        // Deploy new BondToken contract
        BondToken newBondToken = new BondToken(
            string(abi.encodePacked("PYB Series ", seriesId)),
            string(abi.encodePacked("PYB-", seriesId)),
            address(this)
        );
        
        // Transfer ownership of the BondToken to the owner
        newBondToken.transferOwnership(owner());

        bondSeries[seriesId] = BondSeries({
            bondToken: address(newBondToken),
            collateralToken: _collateralToken,
            initialYield: _initialYield,
            totalDeposits: 0,
            active: true
        });

        emit BondSeriesCreated(
            address(newBondToken),
            _collateralToken,
            seriesId,
            _initialYield
        );

        return seriesId;
    }

    function depositCollateral(
        uint256 _seriesId,
        uint256 _amount
    ) external nonReentrant {
        BondSeries storage series = bondSeries[_seriesId];
        require(series.active, "Series not active");
        require(_amount > 0, "Amount must be greater than 0");

        // Transfer collateral from user to this contract
        IERC20(series.collateralToken).transferFrom(
            msg.sender,
            address(this),
            _amount
        );

        // Approve strategy manager to spend tokens
        IERC20(series.collateralToken).approve(address(strategyManager), _amount);

        // Update total deposits
        series.totalDeposits += _amount;

        // Mint bond NFT to user
        BondToken(series.bondToken).mint(
            msg.sender,
            _amount,
            series.initialYield
        );

        // Deploy collateral to strategy
        strategyManager.deployCollateral(
            series.collateralToken,
            _amount,
            _seriesId
        );
    }

    function getBondSeries(uint256 _seriesId) external view returns (
        address bondToken,
        address collateralToken,
        uint256 initialYield,
        uint256 totalDeposits,
        bool active
    ) {
        BondSeries storage series = bondSeries[_seriesId];
        return (
            series.bondToken,
            series.collateralToken,
            series.initialYield,
            series.totalDeposits,
            series.active
        );
    }

    function setStrategyManager(address _strategyManager) external onlyOwner {
        require(_strategyManager != address(0), "Invalid strategy manager address");
        strategyManager = StrategyManager(_strategyManager);
    }
} 