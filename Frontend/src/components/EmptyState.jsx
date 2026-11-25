import { Inbox } from 'lucide-react'

export default function EmptyState({ title = 'No data', message = '' }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-slate-500">
      <Inbox className="h-10 w-10 mb-3 opacity-40" />
      <p className="text-sm font-medium">{title}</p>
      {message && <p className="text-xs mt-1 opacity-70">{message}</p>}
    </div>
  )
}
