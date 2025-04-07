# Perpetual Yield Bonds (PYB) Protocol

A DeFi protocol that lets you earn passive yield on your assets by converting them into tradeable NFT bonds. Simply deposit your assets (ETH, USDC, etc.) and start earning yield immediately!

## How It Works

### 1. Choose Your Bond Series
- Select from available bond series (e.g., USDC with 5% APY, ETH with 4% APY)
- Each series has different:
  - Collateral tokens
  - Yield rates
  - Risk levels

### 2. Deposit Your Assets
1. Connect your wallet
2. Approve the protocol to spend your tokens
3. Choose your deposit amount
4. Receive a bond NFT representing your position

### 3. Earn Yield
Your bond NFT automatically earns yield based on:
- Amount deposited
- Time elapsed
- Series yield rate

You can:
- **Claim Yield**: Get your earned yield in tokens
  - Available anytime
  - Small protocol fee (1%)
  - No lockup period
- **Reinvest Yield**: Add earned yield to your bond
  - Increases your principal
  - Earns compound interest
  - Also subject to protocol fee

### 4. Manage Your Bonds
- View all your bonds in your wallet
- Track yield accumulation
- Transfer or sell your bond NFTs
- Claim or reinvest yield anytime

## Example: Earning Yield with USDC

```plaintext
Initial Deposit:
- Deposit: 1,000 USDC
- Series: Bond Series #1
- Yield Rate: 5% APY
- Get: Bond NFT #123

After 30 days:
- Earned Yield: ~4.17 USDC (5% / 12 months)
- Options:
  a) Claim: Receive ~4.13 USDC (after 1% fee)
  b) Reinvest: Principal becomes 1,004.13 USDC
```

## Benefits

- **Passive Income**: Earn yield without active management
- **Flexibility**: Claim or compound your yield
- **Transparency**: All operations visible on-chain
- **Transferability**: NFTs can be transferred or sold
- **Automation**: Yield generation is automated

## Important Considerations

- Protocol fees apply (1% on yield)
- Yield rates may vary based on market conditions
- Gas fees for transactions
- Smart contract risks exist
- Always do your own research

## Getting Started

1. **Connect Wallet**
   - Use MetaMask or any Web3 wallet
   - Ensure you have tokens to deposit

2. **Choose Bond Series**
   - View available series
   - Check yield rates
   - Select preferred token

3. **Make Deposit**
   - Approve token spending
   - Specify amount
   - Receive bond NFT

4. **Manage Your Bond**
   - View bond details
   - Track yield
   - Choose action:
     - Claim yield
     - Reinvest yield

## Technical Details

For developers and technical users:
- Solidity version: 0.8.19
- Framework: Foundry
- Network: Sepolia testnet (for testing)

## License

MIT
