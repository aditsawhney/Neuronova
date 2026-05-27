import { useState } from 'react'
import axios from 'axios'
import ScoreQuestion from '../components/ScoreQuestion'
import RiskResult from '../components/RiskResult'
import styles from './ScreeningForm.module.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const INATTENTIVE_QUESTIONS = [
  'Has difficulty keeping attention on tasks or activities',
  'Does not seem to listen when spoken to directly',
  'Loses things necessary for tasks or activities',
  'Is easily distracted by unrelated stimuli',
]

const HYPERACTIVE_QUESTIONS = [
  'Fidgets with hands or feet, or squirms in seat',
  'Leaves seat in situations where staying seated is expected',
  'Interrupts or intrudes on others',
  'Has difficulty waiting their turn',
]

const initialScores = () => ({
  inattentive: [null, null, null, null],
  hyperactive: [null, null, null, null],
})

export default function ScreeningForm() {
  const [age, setAge] = useState('')
  const [gender, setGender] = useState('')
  const [handedness, setHandedness] = useState('')
  const [scores, setScores] = useState(initialScores())
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const setScore = (section, index, value) => {
    setScores(prev => {
      const updated = { ...prev, [section]: [...prev[section]] }
      updated[section][index] = value
      return updated
    })
  }

  const inattentiveSum = scores.inattentive.reduce((a, b) => a + (b ?? 0), 0)
  const hyperactiveSum = scores.hyperactive.reduce((a, b) => a + (b ?? 0), 0)

  const allAnswered =
    age && gender && handedness &&
    scores.inattentive.every(v => v !== null) &&
    scores.hyperactive.every(v => v !== null)

  const handleSubmit = async () => {
    if (!allAnswered) return
    if (parseInt(age) < 4 || parseInt(age) > 18) {
      setError('Age must be between 4 and 18.')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const { data } = await axios.post(`${API}/predict`, {
        age: parseFloat(age),
        gender,
        handedness,
        inattentive: inattentiveSum,
        hyper_impulsive: hyperactiveSum,
      })
      setResult(data)
    } catch (err) {
      const detail = err.response?.data?.detail
      if (Array.isArray(detail)) {
        setError('Please check your inputs — age must be between 4 and 18.')
      } else {
        setError(detail || 'Could not reach the screening server. Is the backend running?')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setResult(null)
    setScores(initialScores())
    setAge('')
    setGender('')
    setHandedness('')
    setError(null)
  }

  if (result) {
    return (
      <main className={styles.page}>
        <div className={styles.container}>
          <RiskResult result={result} onReset={handleReset} />
        </div>
      </main>
    )
  }

  return (
    <main className={styles.page}>
      <div className={styles.container}>
        <div className={styles.intro}>
          <h1 className={styles.title}>ADHD Screening</h1>
          <p className={styles.subtitle}>
            Answer the questions below based on the child's behaviour over the past six months.
            This tool provides a referral recommendation - not a diagnosis.
          </p>
        </div>

        {/* Section 1 — basic info */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Basic information</h2>
          <div className={styles.basicGrid}>
            <div className={styles.field}>
              <label className={styles.label}>Age</label>
              <input
                type="number"
                className={styles.input}
                placeholder="e.g. 9"
                min={4}
                max={18}
                value={age}
                onChange={e => setAge(e.target.value)}
              />
            </div>

            <div className={styles.field}>
              <label className={styles.label}>Sex</label>
              <div className={styles.toggleGroup}>
                {['Male', 'Female'].map(g => (
                  <button
                    key={g}
                    type="button"
                    className={gender === g ? `${styles.toggle} ${styles.toggleActive}` : styles.toggle}
                    onClick={() => setGender(g)}
                  >
                    {g}
                  </button>
                ))}
              </div>
            </div>

            <div className={styles.field}>
              <label className={styles.label}>Handedness</label>
              <div className={styles.toggleGroup}>
                {['Left', 'Right', 'Mixed'].map(h => (
                  <button
                    key={h}
                    type="button"
                    className={handedness === h ? `${styles.toggle} ${styles.toggleActive}` : styles.toggle}
                    onClick={() => setHandedness(h)}
                  >
                    {h}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Section 2 — inattention */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Attention behaviours</h2>
          <p className={styles.sectionNote}>How often does the child show the following behaviours?</p>
          <div className={styles.questions}>
            {INATTENTIVE_QUESTIONS.map((q, i) => (
              <ScoreQuestion
                key={i}
                question={q}
                value={scores.inattentive[i]}
                onChange={v => setScore('inattentive', i, v)}
              />
            ))}
          </div>
        </section>

        {/* Section 3 — hyperactivity */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Hyperactivity / impulsivity</h2>
          <p className={styles.sectionNote}>How often does the child show the following behaviours?</p>
          <div className={styles.questions}>
            {HYPERACTIVE_QUESTIONS.map((q, i) => (
              <ScoreQuestion
                key={i}
                question={q}
                value={scores.hyperactive[i]}
                onChange={v => setScore('hyperactive', i, v)}
              />
            ))}
          </div>
        </section>

        {error && (
          <div className={styles.error}>{error}</div>
        )}

        <button
          className={styles.submit}
          onClick={handleSubmit}
          disabled={!allAnswered || loading}
        >
          {loading ? 'Analysing…' : 'Get screening result'}
        </button>
      </div>

      <footer className={styles.footer}>
        Connect with me:<br></br>
        Built by Adit Sawhney · <a href="https://github.com/aditsawhney" target="_blank">GitHub</a> · <a href="https://www.linkedin.com/in/adit-sawhney-ab9175297/" target="_blank">LinkedIn</a>
      </footer>

    </main>
  )
}
