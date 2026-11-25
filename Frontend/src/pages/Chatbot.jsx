/**
 * Section E – AI Security Analyst Chatbot
 * Conversational interface backed by /api/chat.
 */

import { useState, useRef, useEffect, useCallback } from 'react'
import { Bot, Send, User, Lightbulb, AlertCircle } from 'lucide-react'
import { sendChatMessage } from '../services/api'
import Spinner from '../components/Spinner'

const SESSION_ID = `session_${Date.now()}`

const SUGGESTED_QUESTIONS = [
  'What vehicles entered the property today?',
  'Was there any suspicious activity last night?',
  'How many times did the blue Ford F150 appear?',
  'What happened near the garage?',
  'Show me a summary of today\'s security activity.',
  'Were any restricted areas accessed?',
  'Are there any unread alerts?',
]

function Message({ msg }) {
  const isUser = msg.role === 'user'
  const isError = msg.role === 'error'

  if (isError) {
    return (
      <div className="flex items-start gap-3 my-3">
        <div className="shrink-0 mt-0.5 p-1.5 rounded-lg bg-red-900/30">
          <AlertCircle className="h-4 w-4 text-red-400" />
        </div>
        <div className="bg-red-900/20 border border-red-700/30 rounded-xl px-4 py-3 text-sm text-red-300 max-w-2xl">
          {msg.content}
        </div>
      </div>
    )
  }

  return (
    <div className={`flex items-start gap-3 my-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className={`shrink-0 mt-0.5 p-1.5 rounded-lg ${
        isUser ? 'bg-blue-600/30' : 'bg-violet-600/20'
      }`}>
        {isUser
          ? <User className="h-4 w-4 text-blue-400" />
          : <Bot className="h-4 w-4 text-violet-400" />}
      </div>
      <div className={`max-w-2xl rounded-xl px-4 py-3 text-sm leading-relaxed ${
        isUser
          ? 'bg-blue-600/20 border border-blue-600/30 text-blue-100 ml-auto'
          : 'bg-slate-800 border border-slate-700/50 text-slate-200'
      }`}>
        <p className="whitespace-pre-wrap">{msg.content}</p>
        {msg.sources?.length > 0 && (
          <div className="mt-3 pt-3 border-t border-slate-700/50">
            <p className="text-xs text-slate-500 mb-1.5">Sources</p>
            <div className="flex flex-wrap gap-2">
              {msg.sources.map((src, i) => (
                <span key={i} className="text-xs bg-slate-700/50 text-slate-400 px-2 py-0.5 rounded">
                  {src.frame_id ? `Frame #${src.frame_id}` : ''}
                  {src.location ? ` · ${src.location}` : ''}
                  {src.timestamp
                    ? ` · ${new Date(src.timestamp).toLocaleTimeString()}`
                    : ''}
                </span>
              ))}
            </div>
          </div>
        )}
        {msg.timestamp && (
          <p className="text-xs text-slate-600 mt-2">{msg.timestamp}</p>
        )}
      </div>
    </div>
  )
}

export default function Chatbot() {
  const [messages, setMessages] = useState([
    {
      id: 0,
      role: 'assistant',
      content: "Hello! I'm your AI Security Analyst. I can answer questions about your property's drone surveillance footage — vehicles detected, suspicious activities, security alerts, and more. What would you like to know?",
      timestamp: new Date().toLocaleTimeString(),
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)
  const inputRef  = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = useCallback(async (question) => {
    const q = question || input.trim()
    if (!q || loading) return

    setInput('')
    const userMsg = {
      id: Date.now(),
      role: 'user',
      content: q,
      timestamp: new Date().toLocaleTimeString(),
    }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)

    try {
      const res = await sendChatMessage(q, SESSION_ID)
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        content: res.answer,
        sources: res.sources || [],
        timestamp: new Date().toLocaleTimeString(),
      }])
    } catch (e) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'error',
        content: `Error: ${e.message}. Please check that the backend is running and GROQ_API_KEY is set.`,
      }])
    } finally {
      setLoading(false)
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [input, loading])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-200px)] min-h-[500px]">
      {/* Header */}
      <div className="flex items-center gap-3 pb-4 border-b border-slate-700/50">
        <div className="p-2 rounded-lg bg-violet-600/20">
          <Bot className="h-5 w-5 text-violet-400" />
        </div>
        <div>
          <h2 className="text-sm font-semibold text-slate-200">AI Security Analyst</h2>
          <p className="text-xs text-slate-500">Powered by Groq · Grounded in your security data</p>
        </div>
        <span className="ml-auto flex items-center gap-1.5 text-xs text-emerald-400">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
          Online
        </span>
      </div>

      {/* Message area */}
      <div className="flex-1 overflow-y-auto py-4 px-1 space-y-1">
        {messages.map(msg => <Message key={msg.id} msg={msg} />)}
        {loading && (
          <div className="flex items-start gap-3 my-3">
            <div className="shrink-0 mt-0.5 p-1.5 rounded-lg bg-violet-600/20">
              <Bot className="h-4 w-4 text-violet-400" />
            </div>
            <div className="bg-slate-800 border border-slate-700/50 rounded-xl px-4 py-3 flex items-center gap-2">
              <Spinner size="sm" />
              <span className="text-xs text-slate-400">Analysing security data…</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Suggested questions */}
      {messages.length <= 2 && (
        <div className="py-3 border-t border-slate-700/30">
          <div className="flex items-center gap-1.5 mb-2">
            <Lightbulb className="h-3.5 w-3.5 text-yellow-400" />
            <span className="text-xs text-slate-500">Suggested questions</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {SUGGESTED_QUESTIONS.map(q => (
              <button
                key={q}
                onClick={() => send(q)}
                disabled={loading}
                className="text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 hover:border-slate-600 text-slate-400 hover:text-slate-200 px-3 py-1.5 rounded-lg transition-colors disabled:opacity-40"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="pt-3 border-t border-slate-700/50">
        <div className="flex items-end gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about vehicles, alerts, locations, activities…"
            rows={1}
            disabled={loading}
            className="flex-1 bg-slate-800 border border-slate-700 focus:border-blue-500 text-slate-200 text-sm rounded-xl px-4 py-3 resize-none focus:outline-none placeholder:text-slate-500 disabled:opacity-50 leading-relaxed"
            style={{ maxHeight: '120px', overflowY: 'auto' }}
          />
          <button
            onClick={() => send()}
            disabled={!input.trim() || loading}
            className="shrink-0 p-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white transition-colors"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
        <p className="text-xs text-slate-600 mt-1.5 px-1">Press Enter to send · Shift+Enter for newline</p>
      </div>
    </div>
  )
}
