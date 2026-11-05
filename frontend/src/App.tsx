import { useState } from 'react'
import { HistoryPanel } from './components/HistoryPanel'
import { MemoryPanel } from './components/MemoryPanel'
import { SettingsPanel } from './components/SettingsPanel'
import { StatusPanel } from './components/StatusPanel'
import { ToolsPanel } from './components/ToolsPanel'
import { VoicePanel } from './components/VoicePanel'

const TABS = [
  { id: 'voice', label: 'Voice' },
  { id: 'history', label: 'History' },
  { id: 'memory', label: 'Memory' },
  { id: 'tools', label: 'Tools' },
  { id: 'settings', label: 'Settings' },
  { id: 'status', label: 'Status' },
] as const

type TabId = (typeof TABS)[number]['id']

export default function App() {
  const [tab, setTab] = useState<TabId>('voice')
  const [sessionHint, setSessionHint] = useState<string | null>(null)

  return (
    <div className="mx-auto min-h-screen max-w-6xl px-4 py-6">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm uppercase tracking-[0.2em] text-[var(--accent)]">Local · Private · Offline-capable</p>
          <h1 className="font-[family-name:IBM_Plex_Serif,serif] text-4xl font-semibold tracking-tight">
            LocalVoiceAI
          </h1>
          <p className="mt-1 max-w-xl text-[var(--muted)]">
            Real-time voice conversation with local STT, Ollama, Kokoro TTS, and persistent memory.
          </p>
        </div>
        {sessionHint ? (
          <div className="rounded-xl border border-[var(--line)] bg-[var(--bg1)] px-3 py-2 text-xs text-[var(--muted)]">
            Active text session
            <div className="font-mono text-[var(--text)]">{sessionHint.slice(0, 8)}…</div>
          </div>
        ) : null}
      </header>

      <nav className="mb-5 flex flex-wrap gap-2">
        {TABS.map((t) => (
          <button
            key={t.id}
            className="rounded-full px-4 py-2 text-sm transition"
            style={{
              background: tab === t.id ? 'var(--accent)' : 'transparent',
              color: tab === t.id ? '#04140c' : 'var(--muted)',
              border: tab === t.id ? 'none' : '1px solid var(--line)',
            }}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main>
        {tab === 'voice' ? <VoicePanel onSessionHint={setSessionHint} /> : null}
        {tab === 'history' ? <HistoryPanel /> : null}
        {tab === 'memory' ? <MemoryPanel /> : null}
        {tab === 'tools' ? <ToolsPanel /> : null}
        {tab === 'settings' ? <SettingsPanel /> : null}
        {tab === 'status' ? <StatusPanel /> : null}
      </main>
    </div>
  )
}
