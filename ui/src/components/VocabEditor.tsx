import { useState, useEffect } from 'react'

const API_BASE = 'http://localhost:3000/api'

interface VocabRule {
  hear: string[]
  use: string
}

export default function VocabEditor() {
  const [vocab, setVocab] = useState<VocabRule[]>([])
  const [newHear, setNewHear] = useState('')
  const [newUse, setNewUse] = useState('')

  useEffect(() => {
    fetchVocab()
  }, [])

  const fetchVocab = async () => {
    try {
      const response = await fetch(`${API_BASE}/vocab`)
      const data = await response.json()
      setVocab(data.corrections || [])
    } catch (error) {
      console.error('Failed to load vocab', error)
    }
  }

  const saveVocab = async () => {
    try {
      await fetch(`${API_BASE}/vocab`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ corrections: vocab }),
      })
    } catch (error) {
      console.error('Failed to save vocab', error)
    }
  }

  const addRule = () => {
    if (!newHear.trim() || !newUse.trim()) return
    const hear = newHear.split(',').map(s => s.trim()).filter(Boolean)
    setVocab([...vocab, { hear, use: newUse }])
    setNewHear('')
    setNewUse('')
    setTimeout(saveVocab, 100)
  }

  const deleteRule = (index: number) => {
    const updated = vocab.filter((_, i) => i !== index)
    setVocab(updated)
    setTimeout(saveVocab, 100)
  }

  return (
    <section className="setting-section">
      <h2>📚 Custom Vocabulary</h2>
      <p className="description">Teach Voice Inject domain-specific terms</p>
      
      <div className="vocab-list">
        {vocab.map((rule, index) => (
          <div key={index} className="vocab-rule">
            <div className="vocab-hear">
              {rule.hear.join(', ')}
            </div>
            <div className="vocab-arrow">→</div>
            <div className="vocab-use">{rule.use}</div>
            <button onClick={() => deleteRule(index)} className="delete-btn">×</button>
          </div>
        ))}
      </div>

      <div className="vocab-add">
        <input
          type="text"
          placeholder="Variations (comma-separated): y o y, yoy"
          value={newHear}
          onChange={(e) => setNewHear(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && addRule()}
        />
        <input
          type="text"
          placeholder="Correct form: YoY"
          value={newUse}
          onChange={(e) => setNewUse(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && addRule()}
        />
        <button onClick={addRule} className="add-btn">+ Add Rule</button>
      </div>
    </section>
  )
}
