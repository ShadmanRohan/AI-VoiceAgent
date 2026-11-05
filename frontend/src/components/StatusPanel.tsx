import { useEffect, useState } from 'react'
import { api } from '../services/api'
import type { SystemStatus } from '../types'

export function StatusPanel() {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = async () => {
    try {
      const s = await api.status()
      setStatus(s)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  useEffect(() => {
    void refresh()
    const id = setInterval(() => void refresh(), 5000)
    return () => clearInterval(id)
  }, [])

  if (!status) {
    return <p className="text-[var(--muted)]">{error || 'Loading system status…'}</p>
  }

  const hw = status.hardware
  const metrics = (status.metrics?.metrics || {}) as Record<
    string,
    { last?: number; mean?: number; count?: number }
  >

  return (
    <div className="space-y-4">
      <section className="grid gap-3 md:grid-cols-3">
        <Card title="Backend" value={status.backend} ok={status.backend === 'ok'} />
        <Card
          title="Ollama"
          value={status.ollama.ok ? 'running' : 'unavailable'}
          ok={status.ollama.ok}
        />
        <Card
          title="Profile"
          value={String(status.active_profile.profile || hw.recommended_profile)}
          ok
        />
      </section>

      <section className="rounded-2xl border border-[var(--line)] bg-[var(--bg1)]/80 p-4">
        <h2 className="mb-3 text-lg font-semibold">Hardware</h2>
        <dl className="grid gap-2 text-sm md:grid-cols-2">
          <Item k="OS" v={`${hw.os_name} ${hw.architecture}`} />
          <Item k="CPU" v={`${hw.cpu_model} (${hw.cpu_count} threads)`} />
          <Item k="RAM" v={`${hw.ram_available_gb} / ${hw.ram_total_gb} GB available`} />
          <Item
            k="GPU"
            v={
              hw.gpu.available
                ? `${hw.gpu.name} (${hw.gpu.vram_total_mb} MB) CUDA=${hw.gpu.cuda_available}`
                : 'None detected'
            }
          />
        </dl>
        {hw.missing_dependencies.length ? (
          <p className="mt-3 text-sm text-[var(--warn)]">
            Missing: {hw.missing_dependencies.join(', ')}
          </p>
        ) : null}
        {hw.notes.map((n) => (
          <p key={n} className="mt-1 text-sm text-[var(--muted)]">
            {n}
          </p>
        ))}
      </section>

      <section className="rounded-2xl border border-[var(--line)] bg-[var(--bg1)]/80 p-4">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Loaded models / latency</h2>
          <button className="rounded-lg border border-[var(--line)] px-3 py-1 text-sm" onClick={() => void refresh()}>
            Refresh
          </button>
        </div>
        <p className="mb-3 text-sm text-[var(--muted)]">
          Ollama models: {status.ollama.models.join(', ') || 'none'}
        </p>
        <div className="space-y-2 text-sm">
          {Object.keys(metrics).length === 0 ? (
            <p className="text-[var(--muted)]">No latency samples recorded yet.</p>
          ) : (
            Object.entries(metrics).map(([name, m]) => (
              <div key={name} className="flex justify-between gap-3 border-b border-[var(--line)] py-1">
                <span>{name}</span>
                <span className="text-[var(--muted)]">
                  last {m.last?.toFixed?.(1) ?? m.last} ms · mean {m.mean?.toFixed?.(1)} · n={m.count}
                </span>
              </div>
            ))
          )}
        </div>
      </section>
    </div>
  )
}

function Card({ title, value, ok }: { title: string; value: string; ok: boolean }) {
  return (
    <div className="rounded-2xl border border-[var(--line)] bg-[var(--bg1)]/80 p-4">
      <div className="text-xs uppercase tracking-wide text-[var(--muted)]">{title}</div>
      <div className="mt-1 text-xl font-semibold" style={{ color: ok ? 'var(--accent)' : 'var(--danger)' }}>
        {value}
      </div>
    </div>
  )
}

function Item({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <dt className="text-[var(--muted)]">{k}</dt>
      <dd>{v}</dd>
    </div>
  )
}
