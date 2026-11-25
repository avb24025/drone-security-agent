/**
 * Section B – Security Alerts
 * Lists all alerts with severity badges and status controls.
 * Polls every 15 seconds.
 */

import { useEffect, useState, useCallback } from 'react'
import { Bell, CheckCircle, Eye, RefreshCw } from 'lucide-react'
import { getAlerts, updateAlertStatus } from '../services/api'
import Badge from '../components/Badge'
import Card from '../components/Card'
import Spinner from '../components/Spinner'
import EmptyState from '../components/EmptyState'

const SEVERITY_ORDER = { critical: 0, high: 1, medium: 2, low: 3 }

function AlertRow({ alert, onStatusChange }) {
  const [updating, setUpdating] = useState(false)

  const markAs = async (status) => {
    setUpdating(true)
    try {
      await updateAlertStatus(alert.id, status)
      onStatusChange(alert.id, status)
    } catch (e) {
      console.error('Failed to update alert:', e)
    } finally {
      setUpdating(false)
    }
  }

  const typeLabels = {
    suspicious_loitering:       'Loitering',
    unauthorized_vehicle:       'Unauth. Vehicle',
    repeated_vehicle_appearance:'Repeat Vehicle',
    restricted_area_intrusion:  'Restricted Area',
  }

  return (
    <div className="border-b border-slate-700/40 last:border-0 py-4 px-1">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2 mb-1.5">
            <Badge label={alert.severity.toUpperCase()} variant={alert.severity} />
            <Badge label={typeLabels[alert.alert_type] || alert.alert_type} variant="default" />
            <Badge label={alert.status} variant={alert.status} />
          </div>
          <p className="text-sm text-slate-200 leading-snug">{alert.message}</p>
          <div className="flex items-center gap-4 mt-1.5 text-xs text-slate-500">
            <span>{new Date(alert.timestamp).toLocaleString()}</span>
            <span className="text-slate-600">·</span>
            <span>{alert.location}</span>
            {alert.frame_id && <>
              <span className="text-slate-600">·</span>
              <span>Frame #{alert.frame_id}</span>
            </>}
          </div>
        </div>

        {alert.status !== 'resolved' && (
          <div className="flex items-center gap-2 shrink-0">
            {alert.status === 'unread' && (
              <button
                onClick={() => markAs('read')}
                disabled={updating}
                title="Mark as read"
                className="p-1.5 rounded-lg text-slate-400 hover:text-blue-400 hover:bg-blue-500/10 transition-colors disabled:opacity-40"
              >
                <Eye className="h-4 w-4" />
              </button>
            )}
            <button
              onClick={() => markAs('resolved')}
              disabled={updating}
              title="Mark as resolved"
              className="p-1.5 rounded-lg text-slate-400 hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors disabled:opacity-40"
            >
              <CheckCircle className="h-4 w-4" />
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default function Alerts({ refreshTick }) {
  const [alerts, setAlerts] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [severityFilter, setSeverityFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [lastUpdated, setLastUpdated] = useState(null)

  const load = useCallback(async () => {
    try {
      const params = {}
      if (severityFilter) params.severity = severityFilter
      if (statusFilter)   params.status   = statusFilter
      const data = await getAlerts({ ...params, limit: 200 })
      setAlerts(data.sort((a, b) =>
        (SEVERITY_ORDER[a.severity] ?? 9) - (SEVERITY_ORDER[b.severity] ?? 9)
      ))
      setLastUpdated(new Date())
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [severityFilter, statusFilter, refreshTick])

  useEffect(() => { load() }, [load])

  const handleStatusChange = (id, newStatus) => {
    setAlerts(prev => prev.map(a => a.id === id ? { ...a, status: newStatus } : a))
  }

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex gap-2">
          <select
            value={severityFilter}
            onChange={e => setSeverityFilter(e.target.value)}
            className="bg-slate-800 border border-slate-700 text-slate-300 text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue-500"
          >
            <option value="">All severities</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
          <select
            value={statusFilter}
            onChange={e => setStatusFilter(e.target.value)}
            className="bg-slate-800 border border-slate-700 text-slate-300 text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue-500"
          >
            <option value="">All statuses</option>
            <option value="unread">Unread</option>
            <option value="read">Read</option>
            <option value="resolved">Resolved</option>
          </select>
        </div>
        <div className="flex items-center gap-3">
          {lastUpdated && (
            <span className="text-xs text-slate-500">
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={load}
            className="p-1.5 rounded-lg text-slate-400 hover:text-blue-400 hover:bg-blue-500/10 transition-colors"
            title="Refresh"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Alert count */}
      <div className="flex items-center gap-2 text-sm text-slate-400">
        <Bell className="h-4 w-4" />
        <span>{alerts.length} alert{alerts.length !== 1 ? 's' : ''}</span>
      </div>

      {/* Alert list */}
      <Card className="p-0 overflow-hidden">
        {loading ? (
          <div className="flex justify-center py-16"><Spinner /></div>
        ) : error ? (
          <div className="p-5 text-red-400 text-sm">{error}</div>
        ) : alerts.length === 0 ? (
          <EmptyState title="No alerts" message="Process frames to generate security alerts." />
        ) : (
          <div className="divide-y divide-slate-700/30 px-5">
            {alerts.map(alert => (
              <AlertRow key={alert.id} alert={alert} onStatusChange={handleStatusChange} />
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
