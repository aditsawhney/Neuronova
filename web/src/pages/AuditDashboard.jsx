import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts'
import styles from './AuditDashboard.module.css'

const SEX_DATA = [
  { group: 'Male (N=72)', f1: 0.8667, auc: 0.9163 },
  { group: 'Female (N=118)', f1: 0.8850, auc: 0.9468 },
]

const AGE_DATA = [
  { group: '<10 (N=66)', f1: 0.9355, auc: 0.9844 },
  { group: '10–13 (N=75)', f1: 0.9091, auc: 0.9653 },
  { group: '14+ (N=49)', f1: 0.6923, auc: 0.8878 },
]

const IMPORTANCE_DATA = [
  { feature: 'Inattentive', importance: 0.311, std: 0.037 },
  { feature: 'Hyper/Impulsive', importance: 0.175, std: 0.025 },
  { feature: 'Age', importance: 0.017, std: 0.009 },
  { feature: 'Gender', importance: 0.013, std: 0.012 },
  { feature: 'Handedness', importance: 0.002, std: 0.011 },
]

const NAVY = '#0f1f38'
const NAVY_LIGHT = '#2a4a6b'
const AMBER = '#d97706'
const RED = '#991b1b'

function MetricCard({ label, value, sub }) {
  return (
    <div className={styles.metricCard}>
      <span className={styles.metricValue}>{value}</span>
      <span className={styles.metricLabel}>{label}</span>
      {sub && <span className={styles.metricSub}>{sub}</span>}
    </div>
  )
}

function Row({ label, value }) {
  return (
    <div className={styles.row}>
      <dt className={styles.rowLabel}>{label}</dt>
      <dd className={styles.rowValue}>{value}</dd>
    </div>
  )
}

export default function AuditDashboard() {
  return (
    <main className={styles.page}>
      <div className={styles.container}>
        <div className={styles.intro}>
          <h1 className={styles.title}>Model Audit</h1>
          <p className={styles.subtitle}>
            Performance metrics and fairness evaluation on the held-out test set <b>(N=190)</b>.
            All figures are from the final model, never seen during training or validation.
          </p>
        </div>

        {/* Overall metrics */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Overall performance</h2>
          <div className={styles.metricsGrid}>
            <MetricCard label="F1 Score" value="0.881" sub="ADHD class" />
            <MetricCard label="ROC-AUC" value="0.951" />
            <MetricCard label="Precision" value="0.90" sub="ADHD class" />
            <MetricCard label="Recall" value="0.86" sub="ADHD class" />
            <MetricCard label="Accuracy" value="0.91" />
          </div>
        </section>

        {/* Fairness by sex */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Fairness audit — by sex</h2>
          <p className={styles.note}>
            F1 gap between male and female is small <b>(0.018)</b>. ROC-AUC is nearly
            identical, indicating equivalent ranking ability for both groups.
            The F1 difference is explained by fewer ADHD-positive males in the
            test set <b>(22% vs 48%)</b>, not by model bias.
          </p>
          <div className={styles.chartWrap}>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={SEX_DATA} layout="vertical" margin={{ left: 16, right: 32, top: 8, bottom: 8 }}>
                <XAxis type="number" domain={[0, 1]} tickFormatter={v => v.toFixed(1)} tick={{ fontSize: 11, fontFamily: 'DM Mono' }} />
                <YAxis type="category" dataKey="group" width={120} tick={{ fontSize: 12, fontFamily: 'DM Sans' }} />
                <Tooltip formatter={(v) => v.toFixed(4)} />
                <ReferenceLine x={0.85} stroke={AMBER} strokeDasharray="4 3" label={{ value: '0.85', position: 'top', fontSize: 10, fill: AMBER }} />
                <Bar dataKey="f1" name="F1 Score" radius={[0, 4, 4, 0]}>
                  {SEX_DATA.map((_, i) => <Cell key={i} fill={i === 0 ? NAVY_LIGHT : NAVY} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* Fairness by age */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Fairness audit — by age band</h2>
          <p className={styles.note}>
            Performance is strong for children under 14. The adolescent group (14+)
            shows <b>weaker F1 (0.692)</b>, a genuine limitation. The Conners-style behaviour
            scores used as features were primarily normed on children aged 7-13.
          </p>
          <div className={styles.chartWrap}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={AGE_DATA} layout="vertical" margin={{ left: 16, right: 32, top: 8, bottom: 8 }}>
                <XAxis type="number" domain={[0, 1]} tickFormatter={v => v.toFixed(1)} tick={{ fontSize: 11, fontFamily: 'DM Mono' }} />
                <YAxis type="category" dataKey="group" width={120} tick={{ fontSize: 12, fontFamily: 'DM Sans' }} />
                <Tooltip formatter={(v) => v.toFixed(4)} />
                <Bar dataKey="f1" name="F1 Score" radius={[0, 4, 4, 0]}>
                  {AGE_DATA.map((entry, i) => (
                    <Cell key={i} fill={entry.group.startsWith('14') ? RED : NAVY} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* Feature importance */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Feature importance (permutation)</h2>
          <p className={styles.note}>
            Mean decrease in F1 when each feature is randomly shuffled across 10 repeats.
            Inattentive behaviour is the dominant predictor. Handedness contributes
            negligible signal, consistent with ADHD literature on small population-level effects.
          </p>
          <div className={styles.chartWrap}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={IMPORTANCE_DATA} layout="vertical" margin={{ left: 16, right: 32, top: 8, bottom: 8 }}>
                <XAxis type="number" domain={[0, 0.4]} tickFormatter={v => v.toFixed(2)} tick={{ fontSize: 11, fontFamily: 'DM Mono' }} />
                <YAxis type="category" dataKey="feature" width={130} tick={{ fontSize: 12, fontFamily: 'DM Sans' }} />
                <Tooltip formatter={(v) => v.toFixed(4)} />
                <Bar dataKey="importance" name="F1 drop" radius={[0, 4, 4, 0]} fill={NAVY} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* Model info */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Model details</h2>
          <dl className={styles.modelMeta}>
            <Row label="Architecture" value="Soft voting ensemble (RandomForest + ExtraTrees + SVM)" />
            <Row label="Calibration" value="Sigmoid (Platt scaling), cv=5, per sub-estimator" />
            <Row label="Training set" value="567 samples (stratified split)" />
            <Row label="Validation set" value="190 samples (PredefinedSplit)" />
            <Row label="Test set" value="190 samples (held out)" />
            <Row label="Class balance" value="38.2% ADHD / 61.8% control" />
            <Row label="Imbalance handling" value="class_weight='balanced' on all estimators" />
            <Row label="Dataset" value="ADHD-200 phenotypic dataset, multi-site" />
          </dl>
        </section>
      </div>

      <footer className={styles.footer}>
        Contact Me:<br></br>
        Built by Adit Sawhney · <a href="https://github.com/aditsawhney" target="_blank">GitHub</a>
      </footer>
    </main>
  )
}

