/**
 * Section D – Frame Explorer
 */

import { useEffect, useState, useCallback } from 'react'
import { Camera, Play, Layers, Zap, Info, AlertTriangle, CheckCircle, ChevronRight } from 'lucide-react'
import { getFrames, getFrame, getSimulatorFrames, processBatch } from '../services/api'
import Badge from '../components/Badge'
import Card from '../components/Card'
import Spinner from '../components/Spinner'
import EmptyState from '../components/EmptyState'

// ── Frame detail drawer ───────────────────────────────────────────────────────
function FrameDetail({ frameId, onClose }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getFrame(frameId).then(setData).catch(console.error).finally(() => setLoading(false))
  }, [frameId])

  return (
    <>
      <div className="fixed inset-0 z-40 bg-black/50" onClick={onClose} />
      <div className="fixed top-0 right-0 z-50 h-full w-full max-w-lg bg-slate-900 border-l border-slate-700/60 flex flex-col shadow-2xl">
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-700/50 shrink-0">
          <h3 className="text-sm font-semibold text-slate-200">
            {data ? `Frame #${data.frame.frame_id} — ${data.frame.location}` : 'Loading…'}
          </h3>
          <button onClick={onClose} className="text-xs text-slate-400 hover:text-slate-200 px-2 py-1 rounded hover:bg-slate-700/50 transition-colors">
            ✕ Close
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {loading && <div className="flex justify-center py-8"><Spinner /></div>}
          {!loading && !data && <p className="text-sm text-red-400">Could not load frame.</p>}
          {!loading && data && (() => {
            const { frame, events } = data
            const analysis = frame.gemini_analysis || {}
            return (
              <>
                {/* Telemetry grid */}
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {[
                    ['Timestamp', new Date(frame.timestamp).toLocaleString()],
                    ['Location',  frame.location],
                    ['Altitude',  frame.altitude ? `${frame.altitude}m` : '—'],
                    ['GPS',       frame.latitude ? `${frame.latitude.toFixed(4)}, ${frame.longitude.toFixed(4)}` : '—'],
                  ].map(([k, v]) => (
                    <div key={k} className="bg-slate-800/60 rounded p-2.5">
                      <p className="text-slate-500">{k}</p>
                      <p className="text-slate-200 font-medium mt-0.5">{v}</p>
                    </div>
                  ))}
                </div>

                {/* Description */}
                <div className="bg-slate-800/60 rounded p-3">
                  <p className="text-xs text-slate-500 mb-1">Frame Description</p>
                  <p className="text-sm text-slate-300 leading-relaxed">{frame.original_description}</p>
                </div>

                {/* AI Analysis */}
                {analysis.event_type ? (
                  <div className="bg-slate-800/60 rounded p-3 space-y-2">
                    <div className="flex items-center justify-between">
                      <p className="text-xs text-slate-500">AI Analysis</p>
                      <Badge label={analysis.event_type.replace(/_/g, ' ')} variant={analysis.is_suspicious ? 'high' : 'default'} />
                    </div>
                    {analysis.description && <p className="text-sm text-slate-300">{analysis.description}</p>}
                    {analysis.objects?.length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {analysis.objects.map((obj, i) => {
                          const parts = [obj.color, obj.make, obj.model].filter(Boolean)
                          return (
                            <span key={i} className="text-xs bg-slate-700 text-slate-300 px-2 py-0.5 rounded">
                              {obj.type}{parts.length ? ` · ${parts.join(' ')}` : ''}
                            </span>
                          )
                        })}
                      </div>
                    )}
                    {analysis.is_suspicious && (
                      <p className="text-xs text-orange-400 flex items-center gap-1">
                        <AlertTriangle className="h-3 w-3" /> Flagged as suspicious
                      </p>
                    )}
                  </div>
                ) : (
                  <p className="text-xs text-slate-500 flex items-center gap-1.5">
                    <Info className="h-3.5 w-3.5" /> No AI analysis — try reprocessing this frame.
                  </p>
                )}

                {/* Events */}
                {events.length > 0 && (
                  <div>
                    <p className="text-xs text-slate-500 mb-2">Extracted Events ({events.length})</p>
                    <div className="space-y-1.5">
                      {events.map(ev => (
                        <div key={ev.id} className="flex items-center justify-between bg-slate-800/40 rounded px-3 py-2 text-xs">
                          <div className="flex items-center gap-2">
                            <Badge label={ev.object_type} variant={ev.object_type === 'vehicle' ? 'vehicle' : ev.object_type === 'person' ? 'person' : 'default'} />
                            <span className="text-slate-400">{ev.activity || 'detected'}</span>
                          </div>
                          <Badge label={ev.event_type.replace(/_/g, ' ')} variant="default" />
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )
          })()}
        </div>
      </div>
    </>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function FrameExplorer({ refreshTick, onRefresh }) {
  const [simFrames, setSimFrames]   = useState([])
  const [selected, setSelected]     = useState(new Set())
  const [processing, setProcessing] = useState(false)
  const [result, setResult]         = useState(null)
  const [simError, setSimError]     = useState(null)

  const [frames, setFrames]         = useState([])
  const [loadingFrames, setLoadingFrames] = useState(true)
  const [framesError, setFramesError]     = useState(null)
  const [selectedFrame, setSelectedFrame] = useState(null)

  // Load simulator list
  useEffect(() => { getSimulatorFrames().then(setSimFrames).catch(() => {}) }, [])

  // Load indexed frames
  const loadFrames = useCallback(async () => {
    setLoadingFrames(true)
    try {
      const data = await getFrames({ limit: 200 })
      setFrames(data)
      setFramesError(null)
    } catch (e) {
      setFramesError(e.message)
    } finally {
      setLoadingFrames(false)
    }
  }, [refreshTick])

  useEffect(() => { loadFrames() }, [loadFrames])

  const toggleFrame = (id) => setSelected(prev => {
    const next = new Set(prev)
    next.has(id) ? next.delete(id) : next.add(id)
    return next
  })

  const runBatch = async () => {
    if (!selected.size) return
    setProcessing(true); setResult(null); setSimError(null)
    try {
      const res = await processBatch(simFrames.filter(f => selected.has(f.frame_id)))
      setResult(res)
      loadFrames()
      if (onRefresh) onRefresh()
    } catch (e) {
      setSimError(e.message)
    } finally {
      setProcessing(false)
    }
  }

  return (
    <div className="flex flex-col gap-4">

      {/* ── Top section: guide + simulator side by side ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        {/* Guide */}
        <Card className="space-y-3">
          <div className="flex items-center gap-2">
            <Info className="h-4 w-4 text-blue-400 shrink-0" />
            <h3 className="text-sm font-semibold text-slate-200">How it works</h3>
          </div>
          <ol className="space-y-2 text-xs text-slate-400 list-none">
            {[
              'Select frames from the simulator panel.',
              'Click Process — the AI analyzes each frame and extracts security events.',
              'Processed frames appear in the table below. Click any row to inspect.',
              'Check the Alerts and Event History tabs to see what was detected.',
            ].map((text, i) => (
              <li key={i} className="flex gap-2">
                <span className="shrink-0 text-slate-600 font-mono">{i + 1}.</span>
                <span>{text}</span>
              </li>
            ))}
          </ol>

          {/* Scenario reference */}
          <details className="group">
            <summary className="cursor-pointer text-xs text-slate-500 hover:text-slate-300 transition-colors select-none list-none flex items-center gap-1">
              <ChevronRight className="h-3 w-3 group-open:rotate-90 transition-transform" />
              What each frame simulates
            </summary>
            <div className="mt-2 space-y-1">
              {[
                ['1, 9',   'Person loitering at Main Gate at 2 AM',        'high'],
                ['2',      'Blue Ford F150 entering (first time)',          'vehicle'],
                ['3',      'Normal pedestrian — no alert',                  'resolved'],
                ['4',      'Red sedan parked near Garage',                  'medium'],
                ['5',      'Unidentified vehicle entering',                 'high'],
                ['6',      'Person in restricted Garage area',              'critical'],
                ['7, 14',  'Blue Ford F150 reappearing (2nd and 3rd)',      'medium'],
                ['8',      'Routine delivery truck — no alert',             'resolved'],
                ['11',     'Silver SUV, no plates, Side Gate',              'high'],
                ['12, 16', 'Authorised vehicles — no alert',                'resolved'],
                ['13',     'Person in Server Room at 10:30 PM',             'critical'],
                ['17, 20', 'Unknown person during restricted hours',        'high'],
                ['19',     'Routine patrol flyover — no alert',             'resolved'],
              ].map(([f, label, badge]) => (
                <div key={f} className="flex items-center gap-2 text-xs">
                  <span className="font-mono text-slate-600 w-8 shrink-0">#{f}</span>
                  <span className="text-slate-500 flex-1">{label}</span>
                  <Badge label={badge === 'resolved' ? 'safe' : badge} variant={badge} />
                </div>
              ))}
            </div>
          </details>
        </Card>

        {/* Simulator */}
        <Card className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-yellow-400" />
              <h3 className="text-sm font-semibold text-slate-200">Drone Simulator</h3>
            </div>
            <div className="flex gap-3 text-xs">
              <button onClick={() => setSelected(new Set(simFrames.map(f => f.frame_id)))} className="text-blue-400 hover:text-blue-300 transition-colors">All</button>
              <button onClick={() => setSelected(new Set())} className="text-slate-400 hover:text-slate-300 transition-colors">Clear</button>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-1.5 max-h-44 overflow-y-auto">
            {simFrames.map(f => (
              <label key={f.frame_id} className={`flex items-center gap-1.5 px-2 py-1.5 rounded border cursor-pointer text-xs transition-colors
                ${selected.has(f.frame_id) ? 'bg-blue-500/20 border-blue-500/40 text-blue-300' : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-600'}`}>
                <input type="checkbox" checked={selected.has(f.frame_id)} onChange={() => toggleFrame(f.frame_id)} className="sr-only" />
                <span className="font-medium shrink-0">#{f.frame_id}</span>
              </label>
            ))}
          </div>

          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={runBatch}
              disabled={processing || !selected.size}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm rounded-lg font-medium transition-colors"
            >
              {processing ? <Spinner size="sm" /> : <Play className="h-3.5 w-3.5" />}
              Process {selected.size > 0 ? `${selected.size} frame${selected.size > 1 ? 's' : ''}` : 'frames'}
            </button>

            {processing && <span className="text-xs text-blue-400 animate-pulse">Processing…</span>}

            {result && (
              <span className="text-xs text-emerald-400 flex items-center gap-1">
                <CheckCircle className="h-3.5 w-3.5" />
                {result.processed}/{result.total_frames} done · {result.total_events} events · {result.total_alerts} alerts
                {result.failed > 0 && <span className="text-yellow-400 ml-1">· {result.failed} failed</span>}
              </span>
            )}

            {simError && (
              <span className="text-xs text-red-400 flex items-center gap-1">
                <AlertTriangle className="h-3 w-3" />
                {simError.includes('timeout') || simError.includes('429') ? 'Request failed — try again' : simError}
              </span>
            )}
          </div>
        </Card>
      </div>

      {/* ── Indexed frames table ── */}
      <div>
        <div className="flex items-center gap-2 mb-3 text-sm text-slate-400">
          <Layers className="h-4 w-4" />
          <span>{frames.length} indexed frame{frames.length !== 1 ? 's' : ''}</span>
          {frames.length > 0 && <span className="text-xs text-slate-600">— click a row to inspect</span>}
        </div>

        <Card className="p-0 overflow-hidden">
          {loadingFrames ? (
            <div className="flex justify-center py-12"><Spinner /></div>
          ) : framesError ? (
            <p className="p-5 text-sm text-red-400">{framesError}</p>
          ) : frames.length === 0 ? (
            <EmptyState title="No frames yet" message="Select frames in the simulator above and click Process." />
          ) : (
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-slate-700/60">
                  {['Frame', 'Timestamp', 'Location', 'Event Type', 'Suspicious', ''].map(h => (
                    <th key={h} className="px-4 py-2.5 text-xs font-semibold text-slate-500 uppercase tracking-wider whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {frames.map(frame => {
                  const analysis = frame.gemini_analysis || {}
                  return (
                    <tr
                      key={frame.frame_id}
                      onClick={() => setSelectedFrame(frame.frame_id)}
                      className="border-b border-slate-700/30 last:border-0 hover:bg-slate-800/50 cursor-pointer transition-colors"
                    >
                      <td className="px-4 py-3 text-xs font-medium text-slate-300">#{frame.frame_id}</td>
                      <td className="px-4 py-3 text-xs text-slate-400 whitespace-nowrap">{new Date(frame.timestamp).toLocaleString()}</td>
                      <td className="px-4 py-3 text-xs text-slate-400">{frame.location}</td>
                      <td className="px-4 py-3">
                        {analysis.event_type
                          ? <Badge label={analysis.event_type.replace(/_/g, ' ')} variant="default" />
                          : <span className="text-xs text-slate-600">—</span>}
                      </td>
                      <td className="px-4 py-3">
                        {analysis.is_suspicious === true
                          ? <Badge label="yes" variant="high" />
                          : analysis.is_suspicious === false
                            ? <Badge label="no" variant="resolved" />
                            : <span className="text-xs text-slate-600">—</span>}
                      </td>
                      <td className="px-4 py-3 text-xs text-blue-500/60">View →</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </Card>
      </div>

      {/* Drawer */}
      {selectedFrame && (
        <FrameDetail frameId={selectedFrame} onClose={() => setSelectedFrame(null)} />
      )}
    </div>
  )
}
