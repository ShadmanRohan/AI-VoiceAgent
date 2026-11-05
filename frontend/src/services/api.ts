import type {
  AppSettings,
  MemoryItem,
  NoteItem,
  Session,
  SystemStatus,
  TaskItem,
} from '../types'

const BASE = import.meta.env.VITE_API_BASE || ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string }>('/api/health'),
  status: () => request<SystemStatus>('/api/system/status'),
  models: () => request<Record<string, unknown>>('/api/models'),
  getSettings: () => request<AppSettings>('/api/settings'),
  updateSettings: (body: Partial<AppSettings>) =>
    request<AppSettings>('/api/settings', { method: 'PUT', body: JSON.stringify(body) }),

  listSessions: (q?: string) =>
    request<{ sessions: Session[] }>(`/api/sessions${q ? `?q=${encodeURIComponent(q)}` : ''}`),
  getSession: (id: string) => request<Session>(`/api/sessions/${id}`),
  createSession: (title = 'New conversation') =>
    request<Session>('/api/sessions', { method: 'POST', body: JSON.stringify({ title }) }),
  deleteSession: (id: string) =>
    request<{ ok: boolean }>(`/api/sessions/${id}`, { method: 'DELETE' }),

  listMemories: (q?: string) =>
    request<{ consent: boolean; memories: MemoryItem[] }>(
      `/api/memories${q ? `?q=${encodeURIComponent(q)}` : ''}`,
    ),
  createMemory: (content: string, memory_type = 'fact') =>
    request<MemoryItem>('/api/memories', {
      method: 'POST',
      body: JSON.stringify({ content, memory_type }),
    }),
  updateMemory: (id: string, body: Partial<MemoryItem>) =>
    request<MemoryItem>(`/api/memories/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  deleteMemory: (id: string) =>
    request<{ ok: boolean }>(`/api/memories/${id}`, { method: 'DELETE' }),
  setConsent: (consent: boolean) =>
    request<{ consent: boolean }>('/api/memories/consent', {
      method: 'PUT',
      body: JSON.stringify({ consent }),
    }),
  exportMemories: () => request<Record<string, unknown>>('/api/export/memories'),

  listNotes: () => request<{ result?: { notes: NoteItem[] }; notes?: NoteItem[] }>('/api/notes'),
  createNote: (title: string, content: string) =>
    request('/api/notes', { method: 'POST', body: JSON.stringify({ title, content }) }),
  deleteNote: (id: string) =>
    request(`/api/notes/${id}?confirmed=true`, { method: 'DELETE' }),

  listTasks: () => request<{ result?: { tasks: TaskItem[] }; tasks?: TaskItem[] }>('/api/tasks'),
  createTask: (title: string) =>
    request('/api/tasks', { method: 'POST', body: JSON.stringify({ title }) }),
  completeTask: (id: string) =>
    request(`/api/tasks/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ status: 'completed' }),
    }),
  deleteTask: (id: string) => request(`/api/tasks/${id}?confirmed=true`, { method: 'DELETE' }),

  chat: (text: string, session_id?: string) =>
    request<{
      session_id: string
      message_id: string
      text: string
      generation_id: string
    }>('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ text, session_id }),
    }),

  interrupt: (session_id?: string) =>
    request('/api/conversation/interrupt', {
      method: 'POST',
      body: JSON.stringify({ session_id }),
    }),

  ackPlayback: (message_id: string, status = 'spoken') =>
    request('/api/playback/ack', {
      method: 'POST',
      body: JSON.stringify({ message_id, status }),
    }),

  search: (q: string) =>
    request<{ results: Array<Record<string, string>> }>(`/api/search?q=${encodeURIComponent(q)}`),
}
