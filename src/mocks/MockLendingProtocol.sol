// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract MockLendingProtocol is Ownable {
    mapping(address => uint256) public deposits;
    mapping(address => uint256) public lastDepositTime;
    uint256 public interestRatePerSecond = 100; // 0.01% per second (1e4 = 1%)

    IERC20 public token;

    constructor(address _token) {
        token = IERC20(_token);
    }

    function deposit(address from, uint256 amount) external {
        require(amount > 0, "Invalid deposit");

        // Accrue interest before updating
        _accrueInterest(from);

        token.transferFrom(from, address(this), amount);
        deposits[from] += amount;
        lastDepositTime[from] = block.timestamp;
    }

    function withdraw(address to, uint256 amount) external {
        _accrueInterest(to);

        require(deposits[to] >= amount, "Insufficient balance");
        deposits[to] -= amount;
        token.transfer(to, amount);
    }

    function getAccruedBalance(address user) public view returns (uint256) {
        uint256 base = deposits[user];
        uint256 timeElapsed = block.timestamp - lastDepositTime[user];
        uint256 interest = (base * interestRatePerSecond * timeElapsed) / 1e6;
        return base + interest;
    }

    function _accrueInterest(address user) internal {
        if (deposits[user] == 0) return;

        uint256 newAmount = getAccruedBalance(user);
        deposits[user] = newAmount;
        lastDepositTime[user] = block.timestamp;
    }

    // Admin functions
    function setInterestRate(uint256 newRate) external onlyOwner {
        interestRatePerSecond = newRate;
    }

    function recoverTokens(address _to) external onlyOwner {
        uint256 balance = token.balanceOf(address(this));
        token.transfer(_to, balance);
    }
}
