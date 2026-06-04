interface Props {
  value: number
  onChange: (value: number) => void
}

const levels = [
  { value: 1, label: 'Light Touch', desc: 'Minimal cleanup - preserve exact words' },
  { value: 2, label: 'Moderate Edit', desc: 'Add conciseness + tone adjustment' },
  { value: 3, label: 'Heavy Rewrite', desc: 'Full restructuring + heavy editing' },
]

export default function CreativitySlider({ value, onChange }: Props) {
  return (
    <section className="setting-section">
      <h2>🎨 AI Editing Creativity</h2>
      <p className="description">Control how much the AI transforms your dictation</p>
      
      <div className="slider-container">
        <input
          type="range"
          min="1"
          max="3"
          step="1"
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="slider"
        />
        <div className="slider-labels">
          {levels.map((level) => (
            <div
              key={level.value}
              className={`slider-label ${value === level.value ? 'active' : ''}`}
            >
              <strong>{level.label}</strong>
              <span>{level.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
