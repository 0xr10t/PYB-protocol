heres the video for our project
https://vimeo.com/1074681449/aa219e98c6?ts=0&share=copy

# PYB Protocol

Welcome to the **PYB Protocol**! This decentralized finance (DeFi) protocol allows users to deposit collateral, receive yield-generating NFTs, and participate in a robust ecosystem of yield optimization and strategy management.

---

## **Features**

### 1. **Bond NFTs**
- Users deposit collateral (e.g., ETH, USDC) and receive **Bond NFTs**.
- These NFTs represent ownership of the deposited collateral and entitle the holder to claim yield.
- Fully compliant with the ERC721 standard, enabling easy transfer and trading.

### 2. **Yield Generation**
- Collateral is deployed to yield-generating strategies via the **StrategyManager**.
- Yield is distributed periodically to NFT holders through the **YieldDistribution** contract.
- NFT holders can claim or reinvest their yield.

### 3. **Strategy Management**
- The **StrategyManager** optimizes collateral allocation across multiple strategies.
- Supports rebalancing to maintain efficiency and maximize returns.
- Integrates with external lending protocols (e.g., Aave, Compound).

### 4. **Protocol Treasury**
- The **ProtocolTreasury** collects fees and manages reserves.
- Emergency reserves ensure protocol stability.
- Transparent fee collection and reserve management.

### 5. **AI-Powered Optimization**
- AI agents optimize yield strategies, manage risks, and rebalance portfolios.
- Includes:
  - **YieldStrategyAgent**: Maximizes yield across strategies.
  - **RiskManagementAgent**: Monitors and mitigates risks.
  - **RebalancingAgent**: Ensures optimal collateral allocation.
  - **PriceOracleAgent**: Tracks market prices for decision-making.

---

## **How It Works**

### **Step 1: Deposit Collateral**
- Users deposit collateral into the protocol via the `BondFactory` contract.
- A Bond NFT is minted and sent to the user.

### **Step 2: Yield Generation**
- Collateral is deployed to yield-generating strategies by the `StrategyManager`.
- Yield is tracked and distributed by the `YieldDistribution` contract.

### **Step 3: Claim or Reinvest Yield**
- NFT holders can claim their yield or reinvest it to increase the bond's principal amount.

### **Step 4: Trade NFTs**
- Bond NFTs can be sold or transferred, with ownership automatically updated.

---

## **Smart Contracts**

### **1. BondFactory**
- Manages bond series, collateral deposits, and NFT minting.
- Key Functions:
  - `createBondSeries`: Creates a new bond series.
  - `depositCollateral`: Handles user deposits and mints NFTs.

### **2. BondToken**
- Represents Bond NFTs and manages bond metadata.
- Key Functions:
  - `mint`: Mints new Bond NFTs.
  - `claimYield`: Allows NFT holders to claim yield.

### **3. YieldDistribution**
- Manages yield streams and distributes yield to NFT holders.
- Key Functions:
  - `distributeYield`: Updates yield streams.
  - `claimYield`: Transfers earned yield to NFT holders.
  - `reinvestYield`: Reinvests yield into the bond's principal.

### **4. StrategyManager**
- Deploys collateral to yield-generating strategies and handles rebalancing.
- Key Functions:
  - `deployCollateral`: Allocates collateral to strategies.
  - `rebalanceStrategy`: Adjusts collateral allocations.

### **5. ProtocolTreasury**
- Manages protocol fees and reserves.
- Key Functions:
  - `collectFees`: Collects fees from yield distributions.
  - `emergencyWithdraw`: Allows emergency withdrawals.

---

## **AI Agents**

### **1. YieldStrategyAgent**
- Optimizes yield generation across strategies.

### **2. RiskManagementAgent**
- Monitors risks and ensures protocol safety.

### **3. RebalancingAgent**
- Rebalances collateral to maintain efficiency.

### **4. PriceOracleAgent**
- Tracks market prices for informed decision-making.
  
![image](https://github.com/user-attachments/assets/c39f2849-39f8-476d-98ba-6ac49471ac8a)

---

## **Get Started**

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   cd PYB-protocol
   forge install
   cd frontend/PYB
   npm install
   ```
3. Deploy the contracts:
   ```bash
   forge script script/Deploy.s.sol --rpc-url <RPC_URL> --private-key <PRIVATE_KEY>
   ```
4. Run the AI agents:
   ```bash
   cd agentKit/agentkit/python/examples/langchain-cdp-chatbot
   python chatbot.py
   ```

---

## **License**
This project is licensed under the MIT License.
