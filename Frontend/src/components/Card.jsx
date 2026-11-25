/**
 * Generic dark-theme card wrapper.
 */
export default function Card({ children, className = '' }) {
  return (
    <div className={`bg-slate-900 border border-slate-700/50 rounded-xl p-5 ${className}`}>
      {children}
    </div>
  )
}
