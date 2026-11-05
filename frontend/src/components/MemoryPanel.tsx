import { useEffect, useState } from 'react'
import { api } from '../services/api'
import type { MemoryItem } from '../types'

export function MemoryPanel() {
  const [consent, setConsent] = useState(false)
  const [memories, setMemories] = useState<MemoryItem[]>([])
  const [q, setQ] = useState('')
  const [draft, setDraft] = useState('')
  const [editId, setEditId] = useState<string | null>(null)
  const [editText, setEditText] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)

  const load = async (query?: string) => {
    const res = await api.listMemories(query)
    setConsent(res.consent)
    setMemories(res.memories)
  }

  useEffect(() => {
    void load().catch((e) => setError(String(e)))
  }, [])

  const toggleConsent = async () => {
    const res = await api.setConsent(!consent)
    setConsent(res.consent)
  }

  const add = async () => {
    setError(null)
    try {
      await api.createMemory(draft)
      setDraft('')
      await load(q || undefined)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const saveEdit = async () => {
    if (!editId) return
    await api.updateMemory(editId, { content: editText })
    setEditId(null)
    await load(q || undefined)
  }

  const remove = async (id: string) => {
    await api.deleteMemory(id)
    await load(q || undefined)
  }

  const exportData = async () => {
    const data = await api.exportMemories()
    setInfo(`Exported ${Array.isArray(data.memories) ? data.memories.length : 0} memories`)
  }

  return (
    <div className="space-y-4">
      <section className="rounded-2xl border border-[var(--line)] bg-[var(--bg1)]/80 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold">Persistent memory consent</h2>
            <p className="text-sm text-[var(--muted)]">
              Opt-in required before personal facts are stored across sessions.
            </p>
          </div>
          <button
            className="rounded-xl px-4 py-2 font-medium"
            style={{
              background: consent ? 'var(--accent)' : 'transparent',
              color: consent ? '#04140c' : 'var(--text)',
              border: consent ? 'none' : '1px solid var(--line)',
            }}
            onClick={() => void toggleConsent()}
          >
            {consent ? 'Consent enabled' : 'Enable consent'}
          </button>
        </div>
      </section>

      <section className="rounded-2xl border border-[var(--line)] bg-[var(--bg1)]/80 p-4">
        <div className="mb-3 flex flex-wrap gap-2">
          <input
            className="min-w-[220px] flex-1 rounded-xl border border-[var(--line)] bg-[var(--bg2)] px-3 py-2"
            placeholder="Search memories"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && void load(q)}
          />
          <button className="rounded-xl border border-[var(--line)] px-3" onClick={() => void load(q)}>
            Search
          </button>
          <button className="rounded-xl border border-[var(--line)] px-3" onClick={() => void exportData()}>
            Export
          </button>
        </div>

        <div className="mb-4 flex gap-2">
          <input
            className="flex-1 rounded-xl border border-[var(--line)] bg-[var(--bg2)] px-3 py-2"
            placeholder="Add a memory"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
          />
          <button className="rounded-xl bg-[var(--accent)] px-4 py-2 text-black" onClick={() => void add()}>
            Add
          </button>
        </div>

        <div className="space-y-2">
          {memories.map((m) => (
            <div key={m.id} className="rounded-xl border border-[var(--line)] bg-[var(--bg2)] p-3">
              {editId === m.id ? (
                <div className="flex gap-2">
                  <input
                    className="flex-1 rounded-lg border border-[var(--line)] bg-black/20 px-2 py-1"
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                  />
                  <button className="text-[var(--accent)]" onClick={() => void saveEdit()}>
                    Save
                  </button>
                </div>
              ) : (
                <>
                  <div className="text-sm">{m.content}</div>
                  <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-[var(--muted)]">
                    <span>{m.type}</span>
                    <span>importance {m.importance}</span>
                    <button
                      onClick={() => {
                        setEditId(m.id)
                        setEditText(m.content)
                      }}
                    >
                      Edit
                    </button>
                    <button className="text-[var(--danger)]" onClick={() => void remove(m.id)}>
                      Delete
                    </button>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>
        {error ? <p className="mt-3 text-sm text-[var(--danger)]">{error}</p> : null}
        {info ? <p className="mt-3 text-sm text-[var(--accent)]">{info}</p> : null}
      </section>
    </div>
  )
}
