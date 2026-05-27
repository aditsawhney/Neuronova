import { NavLink } from 'react-router-dom'
import styles from './Nav.module.css'

export default function Nav() {
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <div className={styles.brand}>
          <span className={styles.brandName}>NeuroNova</span>
          <span className={styles.brandTag}>ADHD Screening Tool</span>
        </div>
        <nav className={styles.nav}>
          <NavLink
            to="/screen"
            className={({ isActive }) => isActive ? `${styles.link} ${styles.active}` : styles.link}
          >
            Screening
          </NavLink>
          <NavLink
            to="/audit"
            className={({ isActive }) => isActive ? `${styles.link} ${styles.active}` : styles.link}
          >
            Audit
          </NavLink>
          <NavLink
            to="/methodology"
            className={({ isActive }) => isActive ? `${styles.link} ${styles.active}` : styles.link}
          >
            Methodology
          </NavLink>
        </nav>
      </div>
    </header>
  )
}
