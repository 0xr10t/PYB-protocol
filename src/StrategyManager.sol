// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@chainlink/contracts/src/v0.8/automation/AutomationCompatible.sol";

contract StrategyManager is Ownable, ReentrancyGuard, AutomationCompatibleInterface {
    // Events
    event StrategyDeployed(
        address indexed token,
        uint256 amount,
        uint256 seriesId
    );

    event StrategyRebalanced(
        address indexed token,
        uint256 oldAmount,
        uint256 newAmount
    );

    // State variables
    struct Strategy {
        address lendingProtocol;
        uint256 depositedAmount;
        uint256 lastRebalanceTimestamp;
        bool active;
    }

    mapping(address => mapping(uint256 => Strategy)) public strategies;
    mapping(address => bool) public supportedTokens;
    uint256 public rebalanceInterval;
    uint256 public minRebalanceAmount;
    address public bondFactory;

    // Constructor
    constructor(
        uint256 _rebalanceInterval,
        uint256 _minRebalanceAmount
    ) Ownable(msg.sender) {
        rebalanceInterval = _rebalanceInterval;
        minRebalanceAmount = _minRebalanceAmount;
    }

    // Functions
    function setBondFactory(address _bondFactory) external onlyOwner {
        require(_bondFactory != address(0), "Invalid factory address");
        bondFactory = _bondFactory;
    }

    function deployCollateral(
        address token,
        uint256 amount,
        uint256 seriesId
    ) external nonReentrant {
        require(msg.sender == owner() || msg.sender == bondFactory, "Not authorized");
        require(supportedTokens[token], "Token not supported");
        require(amount > 0, "Amount must be greater than 0");

        Strategy storage strategy = strategies[token][seriesId];
        require(strategy.active, "Strategy not active");

        // Transfer tokens from caller to lending protocol
        IERC20(token).transferFrom(msg.sender, strategy.lendingProtocol, amount);
        strategy.depositedAmount += amount;

        emit StrategyDeployed(token, amount, seriesId);
    }

    function rebalanceStrategy(
        address token,
        uint256 seriesId
    ) public onlyOwner nonReentrant {
        Strategy storage strategy = strategies[token][seriesId];
        require(strategy.active, "Strategy not active");
        require(
            block.timestamp >= strategy.lastRebalanceTimestamp + rebalanceInterval,
            "Too early to rebalance"
        );

        uint256 oldAmount = strategy.depositedAmount;
        uint256 newAmount = calculateOptimalAmount(token, seriesId);

        if (newAmount > oldAmount + minRebalanceAmount) {
            // Deploy additional funds
            uint256 additionalAmount = newAmount - oldAmount;
            IERC20(token).transfer(strategy.lendingProtocol, additionalAmount);
            strategy.depositedAmount = newAmount;
        } else if (oldAmount > newAmount + minRebalanceAmount) {
            // Withdraw excess funds
            uint256 withdrawAmount = oldAmount - newAmount;
            strategy.lendingProtocol.call(
                abi.encodeWithSignature(
                    "withdraw(address,uint256)",
                    token,
                    withdrawAmount
                )
            );
            strategy.depositedAmount = newAmount;
        }

        strategy.lastRebalanceTimestamp = block.timestamp;
        emit StrategyRebalanced(token, oldAmount, newAmount);
    }

    function addSupportedToken(address token) external onlyOwner {
        supportedTokens[token] = true;
    }

    function setStrategy(
        address token,
        uint256 seriesId,
        address lendingProtocol
    ) external onlyOwner {
        require(supportedTokens[token], "Token not supported");
        require(lendingProtocol != address(0), "Invalid lending protocol");

        strategies[token][seriesId] = Strategy({
            lendingProtocol: lendingProtocol,
            depositedAmount: 0,
            lastRebalanceTimestamp: block.timestamp,
            active: true
        });
    }

    function calculateOptimalAmount(
        address token,
        uint256 seriesId
    ) public view returns (uint256) {
        // This function would be called by the AI agent to determine optimal amounts
        // For now, returning a simple calculation based on current deposits
        Strategy storage strategy = strategies[token][seriesId];
        return strategy.depositedAmount;
    }

    // Chainlink Automation
    function checkUpkeep(
        bytes calldata /* checkData */
    ) external view override returns (bool upkeepNeeded, bytes memory performData) {
        upkeepNeeded = false;
        performData = "";
        
        // Check if any strategies need rebalancing
        // This is a simplified implementation - in production, you would check all active strategies
        address[3] memory tokens = [address(1), address(2), address(3)]; // Example token addresses
        for (uint256 i = 0; i < 10; i++) { // Check first 10 series
            for (uint256 j = 0; j < 3; j++) { // Check first 3 tokens
                address token = tokens[j];
                Strategy storage strategy = strategies[token][i];
                
                if (strategy.active && 
                    block.timestamp >= strategy.lastRebalanceTimestamp + rebalanceInterval) {
                    upkeepNeeded = true;
                    performData = abi.encode(token, i);
                    break;
                }
            }
            if (upkeepNeeded) break;
        }
        
        return (upkeepNeeded, performData);
    }

    function performUpkeep(bytes calldata performData) external override {
        (address token, uint256 seriesId) = abi.decode(performData, (address, uint256));
        rebalanceStrategy(token, seriesId);
    }
} 