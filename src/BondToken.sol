// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract BondToken is ERC721, Ownable {
    // Events
    event BondMinted(
        address indexed to,
        uint256 indexed tokenId,
        uint256 amount,
        uint256 yieldRate
    );

    event BondAmountUpdated(
        uint256 indexed tokenId,
        uint256 oldAmount,
        uint256 newAmount
    );

    // State variables
    struct Bond {
        uint256 amount;
        uint256 yieldRate;
        uint256 lastClaimTimestamp;
        uint256 unclaimedYield;
        string tokenURI;
    }

    mapping(uint256 => Bond) public bonds;
    uint256 private _tokenIds;
    address public factory;

    // Constructor
    constructor(
        string memory name,
        string memory symbol,
        address _factory
    ) ERC721(name, symbol) Ownable(msg.sender) {
        factory = _factory;
    }

    // Functions
    function mint(
        address to,
        uint256 amount,
        uint256 yieldRate
    ) external onlyFactory {
        _tokenIds++;
        uint256 newTokenId = _tokenIds;

        _safeMint(to, newTokenId);

        bonds[newTokenId] = Bond({
            amount: amount,
            yieldRate: yieldRate,
            lastClaimTimestamp: block.timestamp,
            unclaimedYield: 0,
            tokenURI: ""
        });

        emit BondMinted(to, newTokenId, amount, yieldRate);
    }

    function updateYield(
        uint256 tokenId,
        uint256 newYieldRate
    ) external onlyFactory {
        require(_ownerOf(tokenId) != address(0), "Bond does not exist");
        bonds[tokenId].yieldRate = newYieldRate;
    }

    function updateAmount(
        uint256 tokenId,
        uint256 additionalAmount
    ) external onlyFactory {
        require(_ownerOf(tokenId) != address(0), "Bond does not exist");
        require(additionalAmount > 0, "Amount must be greater than 0");
        
        Bond storage bond = bonds[tokenId];
        uint256 oldAmount = bond.amount;
        bond.amount += additionalAmount;
        
        emit BondAmountUpdated(tokenId, oldAmount, bond.amount);
    }

    function claimYield(uint256 tokenId) external {
        require(ownerOf(tokenId) == msg.sender, "Not bond owner");
        
        Bond storage bond = bonds[tokenId];
        uint256 timeElapsed = block.timestamp - bond.lastClaimTimestamp;
        uint256 yieldEarned = (bond.amount * bond.yieldRate * timeElapsed) / (365 days * 10000);
        
        bond.unclaimedYield += yieldEarned;
        bond.lastClaimTimestamp = block.timestamp;
    }

    function getBondDetails(uint256 tokenId) external view returns (
        uint256 amount,
        uint256 yieldRate,
        uint256 lastClaimTimestamp,
        uint256 unclaimedYield
    ) {
        Bond storage bond = bonds[tokenId];
        return (
            bond.amount,
            bond.yieldRate,
            bond.lastClaimTimestamp,
            bond.unclaimedYield
        );
    }

    // Modifiers
    modifier onlyFactory() {
        require(msg.sender == factory, "Only factory can call this");
        _;
    }

    // URI functions
    function setTokenURI(uint256 tokenId, string memory _tokenURI) external onlyFactory {
        require(_ownerOf(tokenId) != address(0), "URI set of nonexistent token");
        bonds[tokenId].tokenURI = _tokenURI;
    }

    function tokenURI(uint256 tokenId) public view virtual override returns (string memory) {
        require(_ownerOf(tokenId) != address(0), "URI query for nonexistent token");
        return bonds[tokenId].tokenURI;
    }
} 