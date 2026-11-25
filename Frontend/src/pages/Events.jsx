/**
 * Section C – Event History
 * Searchable, filterable table of all extracted events.
 */

import { useEffect, useState, useCallback } from 'react'
import { Search, Filter } from 'lucide-react'
import { getEvents } from '../services/api'
import Badge from '../components/Badge'
import Card from '../components/Card'
import Spinner from '../components/Spinner'
import EmptyState from '../components/EmptyState'

function EventRow({ event }) {
  const attrs = event.object_attributes || {}
  const parts = []
  if (attrs.color) parts.push(attrs.color)
  if (attrs.make)  parts.push(attrs.make)
  if (attrs.model) parts.push(attrs.model)
  const objectLabel = parts.length > 0
    ? `${event.object_type} · ${parts.join(' ')}`
    : event.object_type

  const typeColors = {
    vehicle_entry:              'vehicle',
    vehicle_exit:               'default',
    person_detected:            'person',
    loitering:                  'high',
    restricted_area_intrusion:  'critical',
    unauthorized_vehicle:       'high',
    suspicious_object:          'medium',
    normal_activity:            'resolved',
    unknown:                    'default',
  }

  return (
    <tr className="border-b border-slate-700/40 hover:bg-slate-800/40 transition-colors">
      <td className="py-3 px-4 text-xs text-slate-500">#{event.frame_id}</td>
      <td className="py-3 px-4 text-xs text-slate-400">
        {new Date(event.timestamp).toLocaleString()}
      </td>
      <td className="py-3 px-4">
        <Badge label={event.object_type} variant={event.object_type === 'vehicle' ? 'vehicle' : event.object_type === 'person' ? 'person' : 'default'} />
      </td>
      <td className="py-3 px-4 text-xs text-slate-300">{objectLabel}</td>
      <td className="py-3 px-4">
        <Badge label={event.event_type.replace(/_/g, ' ')} variant={typeColors[event.event_type] || 'default'} />
      </td>
      <td className="py-3 px-4 text-xs text-slate-400">{event.location}</td>
      <td className="py-3 px-4 text-xs text-slate-500 max-w-xs truncate" title={event.activity || ''}>
        {event.activity || '—'}
      </td>
    </tr>
  )
}

export default function Events({ refreshTick }) {
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Filters
  const [objectType, setObjectType] = useState('')
  const [location, setLocation]     = useState('')
  const [eventType, setEventType]   = useState('')

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const params = { limit: 200 }
      if (objectType) params.object_type = objectType
      if (location)   params.location   = location
      if (eventType)  params.event_type = eventType
      const data = await getEvents(params)
      setEvents(data)
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [objectType, location, eventType, refreshTick])

  useEffect(() => { load() }, [load])

  return (
    <div className="space-y-4">
      {/* Filters */}
      <Card className="flex flex-wrap items-center gap-3 py-3">
        <Filter className="h-4 w-4 text-slate-500 shrink-0" />
        <select
          value={objectType}
          onChange={e => setObjectType(e.target.value)}
          className="bg-slate-800 border border-slate-700 text-slate-300 text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue-500"
        >
          <option value="">All object types</option>
          <option value="vehicle">Vehicle</option>
          <option value="person">Person</option>
          <option value="unknown">Unknown</option>
        </select>
        <select
          value={eventType}
          onChange={e => setEventType(e.target.value)}
          className="bg-slate-800 border border-slate-700 text-slate-300 text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue-500"
        >
          <option value="">All event types</option>
          <option value="vehicle_entry">Vehicle Entry</option>
          <option value="vehicle_exit">Vehicle Exit</option>
          <option value="person_detected">Person Detected</option>
          <option value="loitering">Loitering</option>
          <option value="restricted_area_intrusion">Restricted Intrusion</option>
          <option value="unauthorized_vehicle">Unauthorized Vehicle</option>
          <option value="suspicious_object">Suspicious Object</option>
          <option value="normal_activity">Normal Activity</option>
        </select>
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-slate-500" />
          <input
            type="text"
            placeholder="Filter by location…"
            value={location}
            onChange={e => setLocation(e.target.value)}
            className="bg-slate-800 border border-slate-700 text-slate-300 text-sm rounded-lg pl-8 pr-3 py-1.5 w-44 focus:outline-none focus:border-blue-500"
          />
        </div>
        <span className="text-xs text-slate-500 ml-auto">{events.length} event{events.length !== 1 ? 's' : ''}</span>
      </Card>

      {/* Table */}
      <Card className="p-0 overflow-x-auto">
        {loading ? (
          <div className="flex justify-center py-16"><Spinner /></div>
        ) : error ? (
          <div className="p-5 text-red-400 text-sm">{error}</div>
        ) : events.length === 0 ? (
          <EmptyState title="No events" message="Process frames to populate event history." />
        ) : (
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-slate-700/60">
                {['Frame', 'Timestamp', 'Type', 'Object', 'Event', 'Location', 'Activity'].map(h => (
                  <th key={h} className="py-3 px-4 text-xs font-semibold text-slate-500 uppercase tracking-wider whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {events.map(ev => <EventRow key={ev.id} event={ev} />)}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  )
}
