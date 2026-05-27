import styles from './ScoreQuestion.module.css'

const OPTIONS = [
  { label: 'Never', value: 0 },
  { label: 'Sometimes', value: 1 },
  { label: 'Often', value: 2 },
  { label: 'Very Often', value: 3 },
]

export default function ScoreQuestion({ question, value, onChange }) {
  return (
    <div className={styles.row}>
      <p className={styles.question}>{question}</p>
      <div className={styles.options}>
        {OPTIONS.map(opt => (
          <button
            key={opt.value}
            type="button"
            className={value === opt.value ? `${styles.option} ${styles.selected}` : styles.option}
            onClick={() => onChange(opt.value)}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  )
}
