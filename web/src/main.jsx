import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import './index.css'
import Nav from './components/Nav'
import ScreeningForm from './pages/ScreeningForm'
import AuditDashboard from './pages/AuditDashboard'
import Methodology from './pages/Methodology'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <Nav />
      <Routes>
        <Route path="/" element={<Navigate to="/screen" replace />} />
        <Route path="/screen" element={<ScreeningForm />} />
        <Route path="/audit" element={<AuditDashboard />} />
        <Route path="/methodology" element={<Methodology />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>
)
