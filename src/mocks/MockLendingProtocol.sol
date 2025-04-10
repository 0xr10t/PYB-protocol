// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract MockLendingProtocol is Ownable {
    mapping(address => uint256) public deposits;
    mapping(address => uint256) public lastDepositTime;
    uint256 public interestRatePerSecond = 100; // 0.01% per second = 100 (1e4 = 1%)

    IERC20 public token;

    constructor(address _token, address _owner) Ownable(_owner) {
        token = IERC20(_token);
    }

    function deposit(address from, uint256 amount) external {
        require(amount > 0, "Invalid deposit");

        _accrueInterest(from);

        require(token.transferFrom(from, address(this), amount), "Transfer failed");
        deposits[from] += amount;
        lastDepositTime[from] = block.timestamp;
    }

    function withdraw(address to, uint256 amount) external {
        _accrueInterest(to);

        require(deposits[to] >= amount, "Insufficient balance");
        deposits[to] -= amount;

        require(token.transfer(to, amount), "Withdraw transfer failed");
    }

    function getAccruedBalance(address user) public view returns (uint256) {
        uint256 base = deposits[user];
        if (base == 0) return 0;

        uint256 timeElapsed = block.timestamp - lastDepositTime[user];
        uint256 interest = (base * interestRatePerSecond * timeElapsed) / 1e6; // 1e6 since 100 = 0.01%
        return base + interest;
    }

    function _accrueInterest(address user) internal {
        uint256 base = deposits[user];
        if (base == 0) return;

        uint256 accrued = getAccruedBalance(user);
        deposits[user] = accrued;
        lastDepositTime[user] = block.timestamp;
    }

    // Admin
    function setInterestRate(uint256 newRate) external onlyOwner {
        interestRatePerSecond = newRate;
    }

    function recoverTokens(address _to) external onlyOwner {
        uint256 balance = token.balanceOf(address(this));
        require(token.transfer(_to, balance), "Recover failed");
    }
}

