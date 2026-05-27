import styles from './RiskResult.module.css'

const RISK_CONFIG = {
  low: {
    label: 'Low Risk',
    description: 'The behaviour profile described does not currently suggest a high likelihood of ADHD. Continue to monitor and reassess if concerns persist.',
    className: styles.low,
  },
  borderline: {
    label: 'Borderline',
    description: 'The behaviour profile shows some indicators associated with ADHD. Consider monitoring more closely and reassessing in 4-6 weeks.',
    className: styles.borderline,
  },
  high: {
    label: 'Referral Recommended',
    description: 'The behaviour profile is consistent with patterns associated with ADHD. A formal assessment by a qualified clinician is recommended.',
    className: styles.high,
  },
}

export default function RiskResult({ result, onReset }) {
  const config = RISK_CONFIG[result.risk_level]
  const { feature_contributions: fc } = result

  return (
    <div className={styles.wrapper}>
      <div className={`${styles.card} ${config.className}`}>
        <div className={styles.header}>
          <span className={styles.label}>{config.label}</span>
          <span className={styles.prob}>
            {Math.round(result.probability * 100)}% probability
          </span>
        </div>
        <p className={styles.description}>{config.description}</p>
      </div>

      {result.warning && (
        <div className={styles.warning}>
          <span className={styles.warningIcon}>⚠</span>
          <p>{result.warning}</p>
        </div>
      )}

      <div className={styles.contributions}>
        <h3 className={styles.contribTitle}>Score summary</h3>
        <div className={styles.bars}>
          <ScoreBar
            label="Inattention"
            score={fc.inattentive_score}
            max={fc.inattentive_max}
          />
          <ScoreBar
            label="Hyperactivity / Impulsivity"
            score={fc.hyper_impulsive_score}
            max={fc.hyper_impulsive_max}
          />
        </div>
        <div className={styles.meta}>
          <span>{fc.gender}</span>
          <span>Age {fc.age}</span>
        </div>
      </div>

      <p className={styles.disclaimer}>
        This tool provides a <b>referral recommendation</b> only. It is <b>not a diagnosis</b>.
        All results should be interpreted by a qualified professional in the context
        of a full clinical picture.
      </p>

      <button className={styles.reset} onClick={onReset}>
        Start new screening
      </button>
    </div>
  )
}

function ScoreBar({ label, score, max }) {
  const pct = (score / max) * 100
  const isElevated = pct >= 65

  return (
    <div className={styles.barRow}>
      <div className={styles.barLabel}>
        <span>{label}</span>
        <span className={styles.barScore}>{score}/{max}</span>
      </div>
      <div className={styles.barTrack}>
        <div
          className={`${styles.barFill} ${isElevated ? styles.barElevated : ''}`}
          style={{ width: `${pct}%` }}
        />
        <div className={styles.threshold} style={{ left: '65%' }} title="Clinical threshold" />
      </div>
    </div>
  )
}
