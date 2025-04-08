// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract ProtocolTreasury is Ownable, ReentrancyGuard {
    // Events
    event FeesCollected(
        address indexed token,
        uint256 amount
    );

    event EmergencyWithdrawal(
        address indexed token,
        address indexed to,
        uint256 amount
    );

    event ReserveUpdated(
        address indexed token,
        uint256 newAmount
    );

    // State variables
    struct TokenReserves {
        uint256 totalFees;
        uint256 emergencyReserve;
        uint256 lastUpdateTimestamp;
    }

    mapping(address => TokenReserves) public reserves;
    mapping(address => bool) public supportedTokens;
    uint256 public emergencyReserveRatio; // in basis points (1% = 100)
    address public yieldDistribution;
    address public strategyManager;

    constructor(
        address _yieldDistribution,
        address _strategyManager
    ) Ownable(msg.sender) {
        yieldDistribution = _yieldDistribution;
        strategyManager = _strategyManager;
    }

    // Functions
    function collectFees(
        address token,
        uint256 amount
    ) external onlyOwner nonReentrant {
        require(supportedTokens[token], "Token not supported");
        require(amount > 0, "Amount must be greater than 0");

        // Transfer fees to treasury
        IERC20(token).transferFrom(msg.sender, address(this), amount);

        // Update reserves
        TokenReserves storage reserve = reserves[token];
        reserve.totalFees += amount;
        reserve.lastUpdateTimestamp = block.timestamp;

        // Update emergency reserve
        uint256 newReserveAmount = (reserve.totalFees * emergencyReserveRatio) / 10000;
        reserve.emergencyReserve = newReserveAmount;

        emit FeesCollected(token, amount);
        emit ReserveUpdated(token, newReserveAmount);
    }

    function emergencyWithdraw(
        address token,
        address to,
        uint256 amount
    ) external onlyOwner nonReentrant {
        require(supportedTokens[token], "Token not supported");
        require(amount > 0, "Amount must be greater than 0");

        TokenReserves storage reserve = reserves[token];
        require(
            amount <= reserve.totalFees - reserve.emergencyReserve,
            "Amount exceeds available balance"
        );

        // Transfer tokens
        IERC20(token).transfer(to, amount);
        reserve.totalFees -= amount;

        emit EmergencyWithdrawal(token, to, amount);
    }

    function addSupportedToken(address token) external onlyOwner {
        supportedTokens[token] = true;
    }

    function setEmergencyReserveRatio(uint256 _emergencyReserveRatio) external onlyOwner {
        require(_emergencyReserveRatio <= 1000, "Ratio too high"); // Max 10%
        emergencyReserveRatio = _emergencyReserveRatio;
    }

    function getReserves(
        address token
    ) external view returns (
        uint256 totalFees,
        uint256 emergencyReserve,
        uint256 lastUpdateTimestamp
    ) {
        TokenReserves storage reserve = reserves[token];
        return (
            reserve.totalFees,
            reserve.emergencyReserve,
            reserve.lastUpdateTimestamp
        );
    }

    function setYieldDistribution(address _yieldDistribution) external onlyOwner {
        require(_yieldDistribution != address(0), "Invalid yield distribution address");
        yieldDistribution = _yieldDistribution;
    }

    function setStrategyManager(address _strategyManager) external onlyOwner {
        require(_strategyManager != address(0), "Invalid strategy manager address");
        strategyManager = _strategyManager;
    }
}