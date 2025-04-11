import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import metamaskLogo from '../assets/metamask-icon.png'
export function LoginForm({ onConnect }) {
  const [connecting, setConnecting] = useState(false)
  const [account, setAccount] = useState('')
  const [error, setError] = useState('')
  const navigate = useNavigate()
  
  const connectWallet = async () => {
    setConnecting(true)
    setError('')
    
    try {
      if (window.ethereum) {
        const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' })
        setAccount(accounts[0])
        localStorage.setItem('walletAddress', accounts[0])
        setTimeout(() => {
          navigate('/userinfo')
        }, 500)
        if (onConnect) {
          onConnect(accounts[0])
        }
      } else {
        setError('MetaMask is not installed. Please install it to continue.')
      }
    } catch (err) {
      setError(err.message || 'Failed to connect wallet')
    } finally {
      setConnecting(false)
    }
  }
  useEffect(() => {
    const checkConnection = async () => {
      if (window.ethereum) {
        try {
          const accounts = await window.ethereum.request({ method: 'eth_accounts' })
          if (accounts.length > 0) {
            setAccount(accounts[0])
            localStorage.setItem('walletAddress', accounts[0])
            navigate('/userinfo')
          }
        } catch (error) {
          console.error("Error checking connection:", error)
        }
      }
    }
    
    checkConnection()
    if (window.ethereum) {
      window.ethereum.on('accountsChanged', (accounts) => {
        if (accounts.length > 0) {
          setAccount(accounts[0])
          localStorage.setItem('walletAddress', accounts[0])
          navigate('/userinfo')
        } else {
          setAccount('')
          localStorage.removeItem('walletAddress')
        }
      })
    }
    return () => {
      if (window.ethereum && window.ethereum.removeListener) {
        window.ethereum.removeListener('accountsChanged', () => {})
      }
    }
  }, [navigate])
  
  return (
    <div style={{
      backgroundColor: 'rgba(51, 51, 51, 0.9)',
      padding: '30px',
      borderRadius: '10px',
      color: 'white',
      width: '300px',
      textAlign: 'center',
      boxShadow: '0 0 20px rgba(0, 0, 0, 0.5)'
    }}>
      <h2>Login with MetaMask</h2>
      {account ? (
        <div>
          <p>Connected Account:</p>
          <p style={{ fontSize: '14px', wordBreak: 'break-all' }}>{account}</p>
          <button
            onClick={() => {
              setAccount('')
              localStorage.removeItem('walletAddress')
            }}
            style={{
              backgroundColor: '#FF5722',
              padding: '10px 20px',
              borderRadius: '5px',
              border: 'none',
              color: 'white',
              marginTop: '20px',
              cursor: 'pointer'
            }}
          >
            Disconnect
          </button>
        </div>
      ) : (
        <div>
          <p>Connect your wallet to enter the realm</p>
          <div style={{ margin: '20px 0', textAlign: 'center' }}>
            <img src={metamaskLogo} width="80" style={{ display: 'inline-block' }} />
          </div>

          <button
            onClick={connectWallet}
            disabled={connecting}
            style={{
              backgroundColor: '#FF9800',
              padding: '10px 20px',
              borderRadius: '5px',
              border: 'none',
              color: 'white',
              cursor: connecting ? 'default' : 'pointer',
              opacity: connecting ? 0.7 : 1
            }}
          >
            {connecting ? 'Connecting...' : 'Connect Wallet'}
          </button>
          {error && <p style={{ color: '#FF5252', marginTop: '10px' }}>{error}</p>}
        </div>
      )}
    </div>
  )
}
