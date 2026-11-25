/**
 * src/App.jsx
 * -----------
 * Root application shell.
 * - Dark sidebar navigation
 * - Global refresh tick (polls every 15 s)
 * - Backend connectivity indicator
 */

import { useState, useEffect, useCallback } from 'react'
import {
  Activity, AlertTriangle, Camera, Layers,
  MessageSquare, Shield, Wifi, WifiOff,
} from 'lucide-react'
import { getHealth } from './services/api'
import Overview     from './pages/Overview'
import Alerts       from './pages/Alerts'
import Events       from './pages/Events'
import FrameExplorer from './pages/FrameExplorer'
import Chatbot      from './pages/Chatbot'

const NAV_ITEMS = [
  { id: 'overview',  label: 'Overview',       icon: Activity },
  { id: 'alerts',    label: 'Alerts',         icon: AlertTriangle },
  { id: 'events',    label: 'Event History',  icon: Layers },
  { id: 'frames',    label: 'Frame Explorer', icon: Camera },
  { id: 'chat',      label: 'AI Analyst',     icon: MessageSquare },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('overview')
  const [refreshTick, setRefreshTick] = useState(0)
  const [backendOnline, setBackendOnline] = useState(null)   // null = checking

  // Check backend health
  const checkHealth = useCallback(async () => {
    try {
      await getHealth()
      setBackendOnline(true)
    } catch {
      setBackendOnline(false)
    }
  }, [])

  // Auto-refresh every 15 s
  useEffect(() => {
    checkHealth()
    const id = setInterval(() => {
      setRefreshTick(t => t + 1)
      checkHealth()
    }, 15000)
    return () => clearInterval(id)
  }, [checkHealth])

  const forceRefresh = () => setRefreshTick(t => t + 1)

  const pages = {
    overview: <Overview refreshTick={refreshTick} />,
    alerts:   <Alerts   refreshTick={refreshTick} />,
    events:   <Events   refreshTick={refreshTick} />,
    frames:   <FrameExplorer refreshTick={refreshTick} onRefresh={forceRefresh} />,
    chat:     <Chatbot />,
  }

  return (
    <div className="flex h-screen overflow-hidden bg-slate-950">
      {/* ── Sidebar ───────────────────────────────────────────────── */}
      <aside className="w-60 shrink-0 flex flex-col bg-slate-900 border-r border-slate-700/50">
        {/* Logo */}
        <div className="px-5 py-5 border-b border-slate-700/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-blue-600/20">
              <Shield className="h-5 w-5 text-blue-400" />
            </div>
            <div>
              <p className="text-sm font-bold text-slate-100 leading-tight">Drone Security</p>
              <p className="text-xs text-slate-500">Analyst Agent</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === id
                  ? 'bg-blue-600/20 text-blue-400 border border-blue-600/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {label}
            </button>
          ))}
        </nav>

        {/* Backend status */}
        <div className="px-4 py-4 border-t border-slate-700/50">
          <div className={`flex items-center gap-2 text-xs ${
            backendOnline === null ? 'text-slate-500' :
            backendOnline ? 'text-emerald-400' : 'text-red-400'
          }`}>
            {backendOnline === null
              ? <Activity className="h-3.5 w-3.5 animate-pulse" />
              : backendOnline
                ? <Wifi className="h-3.5 w-3.5" />
                : <WifiOff className="h-3.5 w-3.5" />}
            <span>
              {backendOnline === null ? 'Connecting…' :
               backendOnline ? 'Backend connected' : 'Backend offline'}
            </span>
          </div>
          {!backendOnline && backendOnline !== null && (
            <p className="text-xs text-slate-600 mt-1">Start the FastAPI server on port 8000.</p>
          )}
        </div>
      </aside>

      {/* ── Main content ──────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto">
        {/* Page header */}
        <header className="sticky top-0 z-10 bg-slate-950/90 backdrop-blur border-b border-slate-800/60 px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-lg font-semibold text-slate-100">
                {NAV_ITEMS.find(n => n.id === activeTab)?.label}
              </h1>
              <p className="text-xs text-slate-500 mt-0.5">
                Drone Security Analyst · {new Date().toLocaleDateString('en-GB', {
                  weekday: 'long', day: 'numeric', month: 'long', year: 'numeric',
                })}
              </p>
            </div>
            <button
              onClick={forceRefresh}
              className="text-xs text-slate-500 hover:text-blue-400 transition-colors px-3 py-1.5 rounded-lg hover:bg-blue-500/10"
            >
              ↻ Refresh data
            </button>
          </div>
        </header>

        {/* Page body */}
        <div className="px-8 py-6 pb-4">
          {pages[activeTab]}
        </div>
      </main>
    </div>
  )
}
