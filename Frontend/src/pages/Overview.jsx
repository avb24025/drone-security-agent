/**
 * Section A – Overview
 * Displays summary stat cards and monitoring status.
 * Polls /api/dashboard/summary every 15 seconds.
 */

import { useEffect, useState, useCallback } from 'react'
import { Activity, AlertTriangle, Camera, Layers, Shield, TrendingUp } from 'lucide-react'
import { getDashboardSummary, getAlertStats, getEventsSummary } from '../services/api'
import Card from '../components/Card'
import Spinner from '../components/Spinner'

function StatCard({ icon: Icon, label, value, sub, accent = 'blue' }) {
  const accents = {
    blue:   'text-blue-400 bg-blue-500/10',
    red:    'text-red-400 bg-red-500/10',
    orange: 'text-orange-400 bg-orange-500/10',
    violet: 'text-violet-400 bg-violet-500/10',
    emerald:'text-emerald-400 bg-emerald-500/10',
    cyan:   'text-cyan-400 bg-cyan-500/10',
  }
  const cls = accents[accent] || accents.blue
  return (
    <Card className="flex items-start gap-4">
      <div className={`p-2.5 rounded-lg ${cls}`}>
        <Icon className="h-5 w-5" />
      </div>
      <div>
        <p className="text-xs text-slate-500 uppercase tracking-wider">{label}</p>
        <p className="text-2xl font-bold text-white mt-0.5">{value ?? '—'}</p>
        {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
      </div>
    </Card>
  )
}

export default function Overview({ refreshTick }) {
  const [summary, setSummary] = useState(null)
  const [alertStats, setAlertStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const load = useCallback(async () => {
    try {
      const [s, a] = await Promise.all([getDashboardSummary(), getAlertStats()])
      setSummary(s)
      setAlertStats(a)
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load, refreshTick])

  if (loading) return (
    <div className="flex justify-center py-20"><Spinner size="lg" /></div>
  )
  if (error) return (
    <div className="rounded-lg bg-red-900/20 border border-red-700/30 p-4 text-red-400 text-sm">
      Failed to load dashboard: {error}
    </div>
  )

  const isActive = summary?.monitoring_status === 'active'

  return (
    <div className="space-y-6">
      {/* Status banner */}
      <div className={`flex items-center gap-3 rounded-xl px-5 py-3 border ${
        isActive
          ? 'bg-emerald-900/20 border-emerald-700/30 text-emerald-400'
          : 'bg-slate-800/50 border-slate-700/30 text-slate-400'
      }`}>
        <span className={`inline-block h-2.5 w-2.5 rounded-full ${isActive ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'}`} />
        <span className="text-sm font-medium">
          Monitoring {isActive ? 'Active' : 'Idle — process frames to begin'}
        </span>
      </div>

      {/* Stat grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <StatCard icon={Camera}       label="Frames Processed"   value={summary?.total_frames}        accent="blue" />
        <StatCard icon={Layers}       label="Events Extracted"   value={summary?.total_events}        accent="cyan" />
        <StatCard icon={Shield}       label="Total Alerts"       value={summary?.total_alerts}        accent="violet" />
        <StatCard icon={AlertTriangle} label="High Priority"     value={summary?.high_priority_alerts} accent="orange"
          sub="critical + high severity" />
        <StatCard icon={Activity}     label="Unread Alerts"      value={summary?.unread_alerts}       accent="red" />
        <StatCard icon={TrendingUp}   label="Alert Types Active" value={Object.keys(alertStats?.by_type || {}).length} accent="emerald" />
      </div>

      {/* Alert breakdown */}
      {alertStats && alertStats.total > 0 && (
        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Alert Severity Breakdown</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {['critical','high','medium','low'].map(sev => {
              const count = alertStats.by_severity?.[sev] || 0
              const colors = {
                critical: 'bg-red-500',
                high:     'bg-orange-500',
                medium:   'bg-yellow-500',
                low:      'bg-blue-500',
              }
              return (
                <div key={sev} className="bg-slate-800/60 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`inline-block h-2 w-2 rounded-full ${colors[sev]}`} />
                    <span className="text-xs text-slate-400 capitalize">{sev}</span>
                  </div>
                  <p className="text-xl font-bold text-white">{count}</p>
                </div>
              )
            })}
          </div>
        </Card>
      )}
    </div>
  )
}
