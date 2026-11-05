import { useEffect, useState } from 'react'
import { api } from '../services/api'
import type { Session } from '../types'

export function HistoryPanel() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [selected, setSelected] = useState<Session | null>(null)
  const [q, setQ] = useState('')
  const [error, setError] = useState<string | null>(null)

  const load = async (query?: string) => {
    try {
      const res = await api.listSessions(query)
      setSessions(res.sessions)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  useEffect(() => {
    void load()
  }, [])

  const openSession = async (id: string) => {
    const sess = await api.getSession(id)
    setSelected(sess)
  }

  const remove = async (id: string) => {
    await api.deleteSession(id)
    if (selected?.id === id) setSelected(null)
    await load(q || undefined)
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[0.9fr_1.3fr]">
      <section className="rounded-2xl border border-[var(--line)] bg-[var(--bg1)]/80 p-4">
        <div className="mb-3 flex gap-2">
          <input
            className="flex-1 rounded-xl border border-[var(--line)] bg-[var(--bg2)] px-3 py-2"
            placeholder="Search sessions"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void load(q)
            }}
          />
          <button
            className="rounded-xl border border-[var(--line)] px-3"
            onClick={() => void load(q)}
          >
            Search
          </button>
        </div>
        <div className="max-h-[560px] space-y-2 overflow-y-auto">
          {sessions.map((s) => (
            <div
              key={s.id}
              className="flex items-start justify-between gap-2 rounded-xl border border-[var(--line)] bg-[var(--bg2)] p-3"
            >
              <button className="text-left" onClick={() => void openSession(s.id)}>
                <div className="font-medium">{s.title}</div>
                <div className="text-xs text-[var(--muted)]">
                  {s.status} · {s.updated_at || s.started_at}
                </div>
              </button>
              <button
                className="text-xs text-[var(--danger)]"
                onClick={() => void remove(s.id)}
              >
                Delete
              </button>
            </div>
          ))}
        </div>
        {error ? <p className="mt-2 text-sm text-[var(--danger)]">{error}</p> : null}
      </section>

      <section className="rounded-2xl border border-[var(--line)] bg-[var(--bg1)]/80 p-4">
        {!selected ? (
          <p className="text-[var(--muted)]">Open a previous session to read the transcript.</p>
        ) : (
          <>
            <h2 className="mb-1 text-lg font-semibold">{selected.title}</h2>
            <p className="mb-4 text-xs text-[var(--muted)]">
              Resume by continuing in Voice with this context available via history. Session id:{' '}
              {selected.id}
            </p>
            <div className="max-h-[520px] space-y-3 overflow-y-auto">
              {(selected.messages || []).map((m) => (
                <div key={m.id} className="text-sm">
                  <span
                    className="mr-2 font-semibold"
                    style={{ color: m.role === 'USER' ? 'var(--user)' : 'var(--bot)' }}
                  >
                    {m.role === 'USER' ? 'You' : 'AI'}
                  </span>
                  {m.content}
                  {m.interrupted ? (
                    <span className="ml-2 text-xs text-[var(--warn)]">(interrupted)</span>
                  ) : null}
                </div>
              ))}
            </div>
          </>
        )}
      </section>
    </div>
  )
}
