// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "./BondToken.sol";

contract YieldDistribution is Ownable, ReentrancyGuard {
    // Events
    event YieldDistributed(
        address indexed bondToken,
        uint256 indexed tokenId,
        uint256 amount
    );

    event YieldReinvested(
        address indexed bondToken,
        uint256 indexed tokenId,
        uint256 amount
    );

    // State variables
    struct YieldStream {
        uint256 totalYield;
        uint256 lastDistributionTimestamp;
        uint256 yieldPerSecond;
    }

    mapping(address => mapping(uint256 => YieldStream)) public yieldStreams;
    mapping(address => bool) public supportedTokens;
    mapping(address => address) public yieldTokens;
    uint256 public protocolFee; // in basis points (1% = 100)
    address public treasury;
    address public bondFactory;

    // Constructor
    constructor(
        address _treasury,
        address _bondFactory,
        uint256 _protocolFee
    ) Ownable(msg.sender) {
        require(_treasury != address(0), "Invalid treasury address");
        require(_bondFactory != address(0), "Invalid bond factory address");
        require(_protocolFee <= 1000, "Fee too high"); // Max 10%
        treasury = _treasury;
        bondFactory = _bondFactory;
        protocolFee = _protocolFee;
    }

    // Functions
    function setYieldToken(address bondToken, address yieldToken) external onlyOwner {
        require(yieldToken != address(0), "Invalid yield token address");
        yieldTokens[bondToken] = yieldToken;
    }

    function distributeYield(
        address bondToken,
        uint256 tokenId,
        uint256 amount
    ) external onlyOwner nonReentrant {
        require(amount > 0, "Amount must be greater than 0");
        require(yieldTokens[bondToken] != address(0), "Yield token not set");
        
        YieldStream storage stream = yieldStreams[bondToken][tokenId];
        stream.totalYield += amount;
        
        // Calculate yield per second based on the total yield and a 30-day period
        stream.yieldPerSecond = amount / (30 days);
        stream.lastDistributionTimestamp = block.timestamp;
        
        emit YieldDistributed(bondToken, tokenId, amount);
    }

    function claimYield(
        address bondToken,
        uint256 tokenId
    ) external nonReentrant {
        BondToken bond = BondToken(bondToken);
        require(bond.ownerOf(tokenId) == msg.sender, "Not bond owner");
        require(yieldTokens[bondToken] != address(0), "Yield token not set");

        YieldStream storage stream = yieldStreams[bondToken][tokenId];
        uint256 timeElapsed = block.timestamp - stream.lastDistributionTimestamp;
        uint256 yieldEarned = (stream.totalYield * timeElapsed) / (30 days);
        
        require(yieldEarned > 0, "No yield to claim");
        
        // Calculate protocol fee
        uint256 feeAmount = (yieldEarned * protocolFee) / 10000;
        uint256 userAmount = yieldEarned - feeAmount;
        
        // Transfer yield to user
        IERC20(yieldTokens[bondToken]).transfer(msg.sender, userAmount);
        
        // Update last distribution timestamp
        stream.lastDistributionTimestamp = block.timestamp;
        
        emit YieldDistributed(bondToken, tokenId, userAmount);
    }

    function reinvestYield(
        address bondToken,
        uint256 tokenId
    ) external nonReentrant {
        BondToken bond = BondToken(bondToken);
        require(bond.ownerOf(tokenId) == msg.sender, "Not bond owner");

        YieldStream storage stream = yieldStreams[bondToken][tokenId];
        uint256 timeElapsed = block.timestamp - stream.lastDistributionTimestamp;
        uint256 yieldEarned = stream.yieldPerSecond * timeElapsed;
        
        require(yieldEarned > 0, "No yield to reinvest");
        
        // Calculate protocol fee
        uint256 feeAmount = (yieldEarned * protocolFee) / 10000;
        uint256 reinvestAmount = yieldEarned - feeAmount;
        
        // Update bond amount with reinvested yield
        bond.updateAmount(tokenId, reinvestAmount);
        
        emit YieldReinvested(bondToken, tokenId, reinvestAmount);
    }

    function setProtocolFee(uint256 _protocolFee) external onlyOwner {
        require(_protocolFee <= 1000, "Fee too high"); // Max 10%
        protocolFee = _protocolFee;
    }

    function getYieldStream(
        address bondToken,
        uint256 tokenId
    ) external view returns (
        uint256 totalYield,
        uint256 lastDistributionTimestamp,
        uint256 yieldPerSecond
    ) {
        YieldStream storage stream = yieldStreams[bondToken][tokenId];
        return (
            stream.totalYield,
            stream.lastDistributionTimestamp,
            stream.yieldPerSecond
        );
    }

    function setBondFactory(address _bondFactory) external onlyOwner {
        require(_bondFactory != address(0), "Invalid bond factory address");
        bondFactory = _bondFactory;
    }
} 