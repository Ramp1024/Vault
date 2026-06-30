import { useEffect, useState } from 'react'
import './App.css'

type HealthResponse = {
  status: string
}

function App() {
  const [status, setStatus] = useState<string>('loading')

  useEffect(() => {
    let active = true

    const loadHealth = async (): Promise<void> => {
      try {
        const response = await fetch('/api/health')
        if (!response.ok) {
          throw new Error(`Request failed with ${response.status}`)
        }

        const data: HealthResponse = await response.json()
        if (active) {
          setStatus(data.status)
        }
      } catch {
        if (active) {
          setStatus('unreachable')
        }
      }
    }

    void loadHealth()

    return () => {
      active = false
    }
  }, [])

  return (
    <main className="status-page">
      <h1>Backend Health</h1>
      <p className="status-text">status: {status}</p>
    </main>
  )
}

export default App
