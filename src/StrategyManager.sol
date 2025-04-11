// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@chainlink/contracts/src/v0.8/automation/AutomationCompatible.sol";
import "../src/mocks/MockLendingProtocol.sol";

contract StrategyManager is
    Ownable,
    ReentrancyGuard,
    AutomationCompatibleInterface
{
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

    struct Strategy {
        address lendingProtocol;
        uint256 depositedAmount;
        uint256 lastRebalanceTimestamp;
        bool active;
    }

    mapping(address => mapping(uint256 => Strategy)) public strategies;
    mapping(address => bool) public supportedTokens;
    address[] public supportedTokenList; // New array to track added tokens
    uint256 public rebalanceInterval;
    uint256 public minRebalanceAmount;
    address public bondFactory;

    constructor(
        uint256 _rebalanceInterval,
        uint256 _minRebalanceAmount
    ) Ownable(msg.sender) {
        rebalanceInterval = _rebalanceInterval;
        minRebalanceAmount = _minRebalanceAmount;
    }

    function setBondFactory(address _bondFactory) external onlyOwner {
        require(_bondFactory != address(0), "Invalid factory address");
        bondFactory = _bondFactory;
    }

    function deployCollateral(
        address token,
        uint256 amount,
        uint256 seriesId
    ) external nonReentrant {
        require(
            msg.sender == owner() || msg.sender == bondFactory,
            "Not authorized"
        );
        require(supportedTokens[token], "Token not supported");
        require(amount > 0, "Amount must be greater than 0");

        Strategy storage strategy = strategies[token][seriesId];
        require(strategy.active, "Strategy not active");

        IERC20(token).transferFrom(msg.sender, address(this), amount);
        IERC20(token).approve(strategy.lendingProtocol, amount);

        MockLendingProtocol(strategy.lendingProtocol).deposit(
            address(this),
            amount
        );
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
        block.timestamp >=
            strategy.lastRebalanceTimestamp + rebalanceInterval,
        "Too early to rebalance"
    );

    MockLendingProtocol protocol = MockLendingProtocol(
        strategy.lendingProtocol
    );

    // Accrue interest before calculating balances
    protocol.withdraw(address(this), 0); // Trigger interest accrual

    uint256 accruedAmount = protocol.getAccruedBalance(address(this));
    uint256 oldAmount = strategy.depositedAmount;

    if (accruedAmount > oldAmount + minRebalanceAmount) {
        uint256 profit = accruedAmount - oldAmount;
        protocol.withdraw(address(this), profit);
        IERC20(token).transfer(owner(), profit); // send yield to owner or treasury
    }

    strategy.lastRebalanceTimestamp = block.timestamp;
    emit StrategyRebalanced(token, oldAmount, accruedAmount);
    strategy.depositedAmount = accruedAmount;
}

    function addSupportedToken(address token) external onlyOwner {
        if (!supportedTokens[token]) {
            supportedTokens[token] = true;
            supportedTokenList.push(token); // add to list for upkeep loop
        }
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
        Strategy storage strategy = strategies[token][seriesId];
        return strategy.depositedAmount;
    }

    function checkUpkeep(
        bytes calldata
    )
        external
        view
        override
        returns (bool upkeepNeeded, bytes memory performData)
    {
        upkeepNeeded = false;
        performData = "";

        for (uint256 i = 0; i < supportedTokenList.length; i++) {
            address token = supportedTokenList[i];
            for (uint256 j = 0; j < 10; j++) {
                Strategy storage strategy = strategies[token][j];
                if (
                    strategy.active &&
                    block.timestamp >=
                    strategy.lastRebalanceTimestamp + rebalanceInterval
                ) {
                    upkeepNeeded = true;
                    performData = abi.encode(token, j);
                    return (upkeepNeeded, performData);
                }
            }
        }
    }

    function performUpkeep(bytes calldata performData) external override {
        (address token, uint256 seriesId) = abi.decode(
            performData,
            (address, uint256)
        );
        rebalanceStrategy(token, seriesId);
    }
}
