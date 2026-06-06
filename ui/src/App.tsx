import { useState, useEffect } from 'react'
import CreativitySlider from './components/CreativitySlider'
import ToneSelector from './components/ToneSelector'
import VocabEditor from './components/VocabEditor'
import UserContextEditor from './components/UserContextEditor'
import './App.css'

const API_BASE = 'http://localhost:3000/api'

interface Config {
  creativity_level: number
  tone: string
  model: string
  region: string
}

function App() {
  const [config, setConfig] = useState<Config | null>(null)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')

  useEffect(() => {
    fetchConfig()
  }, [])

  const fetchConfig = async () => {
    try {
      const response = await fetch(`${API_BASE}/config`)
      const data = await response.json()
      setConfig(data)
    } catch (error) {
      setMessage('⚠️ Failed to load config. Is the API server running?')
    }
  }

  const saveConfig = async () => {
    if (!config) return
    setSaving(true)
    try {
      await fetch(`${API_BASE}/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      setMessage('✅ Settings saved! Changes apply immediately.')
      setTimeout(() => setMessage(''), 3000)
    } catch (error) {
      setMessage('❌ Failed to save settings')
    } finally {
      setSaving(false)
    }
  }

  if (!config) {
    return (
      <div className="app">
        <div className="loading">Loading settings...</div>
      </div>
    )
  }

  return (
    <div className="app">
      <header>
        <h1>🎙️ Voice Inject Settings</h1>
        <p>Configure AI editing creativity, tone, and custom vocabulary</p>
      </header>

      <main>
        <UserContextEditor />

        <CreativitySlider
          value={config.creativity_level}
          onChange={(value) => setConfig({ ...config, creativity_level: value })}
        />

        <ToneSelector
          value={config.tone}
          onChange={(value) => setConfig({ ...config, tone: value })}
        />

        <VocabEditor />

        <button onClick={saveConfig} disabled={saving} className="save-btn">
          {saving ? 'Saving...' : '💾 Save Settings'}
        </button>

        {message && <div className="message">{message}</div>}
      </main>

      <footer>
        <p>Changes apply immediately - no restart needed!</p>
        <p className="voice-hint">
          💡 Next time, just type <code>voice</code> in your terminal to launch this app.
        </p>
      </footer>
    </div>
  )
}

export default App
