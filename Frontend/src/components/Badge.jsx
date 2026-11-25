/**
 * Severity / status badge used across the dashboard.
 */
export default function Badge({ label, variant = 'default', className = '' }) {
  const variants = {
    critical: 'bg-red-500/20 text-red-400 border border-red-500/30',
    high:     'bg-orange-500/20 text-orange-400 border border-orange-500/30',
    medium:   'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30',
    low:      'bg-blue-500/20 text-blue-400 border border-blue-500/30',
    unread:   'bg-violet-500/20 text-violet-400 border border-violet-500/30',
    read:     'bg-slate-500/20 text-slate-400 border border-slate-500/30',
    resolved: 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30',
    vehicle:  'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30',
    person:   'bg-purple-500/20 text-purple-400 border border-purple-500/30',
    default:  'bg-slate-700/50 text-slate-300 border border-slate-600/50',
  }
  const cls = variants[variant] || variants.default
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${cls} ${className}`}>
      {label}
    </span>
  )
}
