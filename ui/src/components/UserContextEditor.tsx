import { useState, useEffect } from 'react'

const API_BASE = 'http://localhost:3000/api'

export default function UserContextEditor() {
  const [context, setContext] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')

  useEffect(() => {
    fetchContext()
  }, [])

  const fetchContext = async () => {
    try {
      const response = await fetch(`${API_BASE}/user-context`)
      const data = await response.json()
      setContext(data.user_context)
    } catch (error) {
      setMessage('⚠️ Failed to load user context')
    } finally {
      setLoading(false)
    }
  }

  const saveContext = async () => {
    setSaving(true)
    try {
      await fetch(`${API_BASE}/user-context`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_context: context }),
      })
      setMessage('✅ User context saved! This is cached for faster AI responses.')
      setTimeout(() => setMessage(''), 3000)
    } catch (error) {
      setMessage('❌ Failed to save user context')
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return <div className="section">Loading user context...</div>
  }

  return (
    <div className="section">
      <h2>👤 User Context (Cached Prompt)</h2>
      <p className="description">
        This describes your professional background to help the AI understand technical terms.
        This is cached with your vocabulary for faster responses (~90% cost reduction).
      </p>

      <textarea
        value={context}
        onChange={(e) => setContext(e.target.value)}
        placeholder="Data scientist and software developer working with AWS, Python, and AI/ML..."
        rows={6}
        style={{
          width: '100%',
          padding: '12px',
          fontSize: '14px',
          fontFamily: 'monospace',
          borderRadius: '8px',
          border: '2px solid #ddd',
          resize: 'vertical',
          marginTop: '8px'
        }}
      />

      <button 
        onClick={saveContext} 
        disabled={saving}
        style={{
          marginTop: '12px',
          padding: '10px 20px',
          fontSize: '14px',
          fontWeight: 600,
          backgroundColor: '#4CAF50',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          cursor: saving ? 'not-allowed' : 'pointer',
          opacity: saving ? 0.6 : 1
        }}
      >
        {saving ? 'Saving...' : '💾 Save User Context'}
      </button>

      {message && (
        <div style={{ 
          marginTop: '12px', 
          padding: '10px', 
          borderRadius: '6px',
          backgroundColor: message.includes('✅') ? '#d4edda' : '#f8d7da',
          color: message.includes('✅') ? '#155724' : '#721c24',
          fontSize: '14px'
        }}>
          {message}
        </div>
      )}

      <div style={{ 
        marginTop: '16px', 
        padding: '12px', 
        backgroundColor: '#e7f3ff', 
        borderRadius: '6px',
        fontSize: '13px',
        color: '#0056b3'
      }}>
        💡 <strong>Tip:</strong> Include your role, frequently used tools, and domain-specific terms.
        The AI will use this context to correctly interpret technical language.
      </div>
    </div>
  )
}
