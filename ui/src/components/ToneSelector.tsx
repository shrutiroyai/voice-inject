interface Props {
  value: string
  onChange: (value: string) => void
}

const tones = [
  { value: 'casual', label: 'Casual', emoji: '😊', desc: 'Friendly, conversational' },
  { value: 'professional', label: 'Professional', emoji: '💼', desc: 'Formal, polished' },
  { value: 'excited', label: 'Excited', emoji: '🎉', desc: 'Energetic, enthusiastic' },
  { value: 'neutral', label: 'Neutral', emoji: '📝', desc: 'Balanced, matter-of-fact' },
]

export default function ToneSelector({ value, onChange }: Props) {
  return (
    <section className="setting-section">
      <h2>🎭 Default Tone</h2>
      <p className="description">Choose the default writing style for your dictation</p>
      
      <div className="tone-grid">
        {tones.map((tone) => (
          <button
            key={tone.value}
            onClick={() => onChange(tone.value)}
            className={`tone-button ${value === tone.value ? 'active' : ''}`}
          >
            <span className="tone-emoji">{tone.emoji}</span>
            <strong>{tone.label}</strong>
            <span className="tone-desc">{tone.desc}</span>
          </button>
        ))}
      </div>
    </section>
  )
}
