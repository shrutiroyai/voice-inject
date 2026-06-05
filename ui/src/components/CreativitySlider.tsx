interface Props {
  value: number
  onChange: (value: number) => void
}

export default function CreativitySlider({ value, onChange }: Props) {
  return (
    <section className="setting-section">
      <h2>🎨 AI Temperature</h2>
      <p className="description">
        Control how creative the AI is (0 = conservative, 1 = creative)
      </p>
      
      <div className="slider-container">
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="slider"
        />
        <div className="slider-value">
          <strong>Temperature: {value.toFixed(1)}</strong>
          <span className="temp-desc">
            {value <= 0.2 ? 'Very Conservative - Minimal changes' :
             value <= 0.4 ? 'Conservative - Light editing' :
             value <= 0.6 ? 'Balanced - Moderate editing' :
             value <= 0.8 ? 'Creative - Significant rewriting' :
             'Very Creative - Heavy transformation'}
          </span>
        </div>
      </div>
    </section>
  )
}
