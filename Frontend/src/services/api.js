/**
 * src/services/api.js
 * -------------------
 * Centralised Axios client for the Drone Security Agent backend.
 * All components import from here — the base URL is configured once.
 */

import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

// ── Response interceptor — normalise errors ──────────────────────────────────
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const message =
      err.response?.data?.detail ||
      err.response?.statusText ||
      err.message ||
      'Unknown error'
    return Promise.reject(new Error(message))
  }
)

// ── Dashboard ────────────────────────────────────────────────────────────────
export const getDashboardSummary = () =>
  api.get('/api/dashboard/summary').then((r) => r.data)

// ── Frames ───────────────────────────────────────────────────────────────────
export const getFrames = (params = {}) =>
  api.get('/api/frames', { params }).then((r) => r.data)

export const getFrame = (frameId) =>
  api.get(`/api/frames/${frameId}`).then((r) => r.data)

export const getSimulatorFrames = () =>
  api.get('/api/frames/simulator/frames').then((r) => r.data)

export const processFrame = (frame) =>
  api.post('/api/frames/process', frame).then((r) => r.data)

export const processBatch = (frames) =>
  api.post('/api/frames/batch', { frames }).then((r) => r.data)

// ── Events ───────────────────────────────────────────────────────────────────
export const getEvents = (params = {}) =>
  api.get('/api/events', { params }).then((r) => r.data)

export const getEventsSummary = () =>
  api.get('/api/events/summary').then((r) => r.data)

// ── Alerts ───────────────────────────────────────────────────────────────────
export const getAlerts = (params = {}) =>
  api.get('/api/alerts', { params }).then((r) => r.data)

export const getAlertStats = () =>
  api.get('/api/alerts/stats').then((r) => r.data)

export const updateAlertStatus = (alertId, status) =>
  api.patch(`/api/alerts/${alertId}`, { status }).then((r) => r.data)

// ── Chat ─────────────────────────────────────────────────────────────────────
export const sendChatMessage = (question, sessionId = 'default') =>
  api.post('/api/chat', { question, session_id: sessionId }).then((r) => r.data)

// ── Health ───────────────────────────────────────────────────────────────────
export const getHealth = () =>
  api.get('/health').then((r) => r.data)
