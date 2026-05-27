import styles from './Methodology.module.css'

export default function Methodology() {
  return (
    <main className={styles.page}>
      <div className={styles.container}>
        <div className={styles.intro}>
          <h1 className={styles.title}>Methodology</h1>
          <p className={styles.subtitle}>
            How NeuroNova works, what data it uses, and where its limits lie.
          </p>
        </div>

        <Section title="What this tool does">
          <p>
            NeuroNova is a <b>pre-assessment screening tool</b>. It takes information a
            school counsellor, GP, or parent would know before any clinical tests
            are run, and returns a referral recommendation - <b>not a diagnosis</b>.
          </p>
          <p>
            The goal is to reduce the time between a parent or teacher noticing
            concerning behaviour and a formal clinical assessment being initiated,
            particularly for groups that are historically underreferred: girls and
            younger children.
          </p>
        </Section>

        <Section title="What it is not">
          <p>
            This tool <b>cannot diagnose ADHD</b>. A formal diagnosis requires a full
            clinical assessment by a qualified professional, including standardised
            testing, observation across multiple settings, and ruling out alternative
            explanations.
          </p>
          <p>
            A "high risk" result means the behaviour profile is statistically
            consistent with patterns seen in children who receive an ADHD diagnosis.
            It does not mean the child has ADHD. A "low risk" result does not rule
            out ADHD - particularly in girls, where presentations are often subtler.
          </p>
        </Section>

        <Section title="Data source">
          <p>
            The model was trained on the <b>ADHD-200 phenotypic dataset</b> - a multi-site
            research dataset of <b>973 participants</b>(children and adolescents) with
            confirmed diagnoses or typically-developing status. After removing
            participants with incomplete or ambiguous diagnostic labels, <b>947</b> rows
            were retained.
          </p>
          <p>
            <b>Class balance: 38.2% ADHD, 61.8% typically developing.</b>
          </p>
        </Section>

        <Section title="Features used">
          <p>
            Only information available before any clinical assessment was retained:
          </p>
          <ul className={styles.list}>
            <li><strong>Age:</strong> child's age at screening</li>
            <li><strong>Sex:</strong> male or female</li>
            <li><strong>Handedness:</strong> left, right, or mixed</li>
            <li><strong>Inattention score:</strong> sum of four Conners-style behaviour items (0–12)</li>
            <li><strong>Hyperactivity/impulsivity score:</strong> sum of four Conners-style behaviour items (0–12)</li>
          </ul>
          <p>
            IQ, medication status, formal clinical scores, and secondary diagnoses
            were all excluded - these require clinical input and would not be
            available at the point of referral.
          </p>
        </Section>

        <Section title="Score interpretation">
          <p>
            The four behaviour questions in each section are scored:
          </p>
          <ul className={styles.list}>
            <li>Never (0),</li>
            <li>Sometimes (1),</li>
            <li>Often (2),</li>
            <li>Very Often (3). </li>
          </ul>
          <p>  
            The sum (0-12) is linearly mapped to the <b>Conners T-score scale (9-90)</b> used in the training dataset,
            where scores above 65 are considered clinically elevated.
          </p>
          <p>
            The model outputs a probability, which is mapped to three risk levels:
          </p>
            <ul className={styles.list}>
              <li>low (&lt;35%),</li>
              <li>borderline (35-65%),</li>
              <li>and high (&gt;65%).</li>
            </ul>
          <p>
            The borderline band is deliberately wide. Calibration analysis showed that mid-range
            probability estimates are less reliable than high or low predictions on
            this sample size.
          </p>
        </Section>

        <Section title="Model">
          <p>
            A <b>soft-voting ensemble</b> of three individually calibrated classifiers:
            <b>Random Forest, Extra Trees, and a radial-basis SVM</b>. Each was tuned via 
            <b> grid search</b> on a fixed validation fold using <b>F1</b> as the scoring metric.
            <b>Calibration (Platt scaling) </b>was applied to each sub-estimator individually
            to preserve the preprocessing pipeline at inference time.
          </p>
          <p>
            Test set performance (N=190, never seen during training or tuning):<br></br>
            <b>F1 0.881, ROC-AUC 0.951, accuracy 0.91.</b>
          </p>
        </Section>

        <Section title="Known limitations">
          <ul className={styles.list}>
            <li>
              <strong>Adolescents (14+):</strong> model F1 drops to 0.692 for this
              age group. The behaviour rating scales were normed on younger children,
              and ADHD presentations change with age. Results for children 14 and
              over should be treated with additional caution.
            </li>
            <li>
              <strong>Dataset size:</strong> 947 participants from a research setting
              is small for a machine learning model. Predictions may not generalise
              equally to all populations, schools, or cultural contexts.
            </li>
            <li>
              <strong>Self-report bias:</strong> behaviour scores depend on the
              accuracy and consistency of the person completing the form. The same
              child may score differently depending on who answers.
            </li>
            <li>
              <strong>Handedness:</strong> included in the form for completeness,
              but contributes negligible predictive weight in this dataset (permutation
              importance ≈ 0).
            </li>
            <li>
              <strong>Not a replacement for clinical judgement:</strong> this tool
              is a decision support aid. Clinical judgement, context, and the full
              history of a child should always take precedence.
            </li>
          </ul>
        </Section>

        <Section title="Fairness">
          <p>
            The model was audited for differential performance by sex and age band.
            Male/female F1 gap is 0.018 (0.867 vs 0.885); ROC-AUC is nearly
            identical (0.916 vs 0.947), indicating <b>no meaningful sex bias in
            discriminative ability</b>.
          </p>
          <p>
            The adolescent performance gap is documented above and surfaced as a
            warning in the tool output for all children aged 14 and over.
          </p>
        </Section>
      </div>

      <footer className={styles.footer}>
        Contact Me:<br></br>
        Built by Adit Sawhney · <a href="https://github.com/aditsawhney" target="_blank">GitHub</a>
      </footer>
    </main>
  )
}

function Section({ title, children }) {
  return (
    <section className={styles.section}>
      <h2 className={styles.sectionTitle}>{title}</h2>
      <div className={styles.body}>{children}</div>
    </section>
  )
}
