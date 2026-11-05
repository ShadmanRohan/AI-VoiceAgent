import { useEffect, useState, type ReactNode } from 'react'
import { api } from '../services/api'
import type { AppSettings } from '../types'

const VOICES = [
  'af_heart',
  'af_bella',
  'af_sarah',
  'am_adam',
  'am_michael',
  'bf_emma',
  'bf_isabella',
  'bm_george',
  'bm_lewis',
]

export function SettingsPanel() {
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [models, setModels] = useState<string[]>([])
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    void (async () => {
      try {
        const [s, m] = await Promise.all([api.getSettings(), api.models()])
        setSettings(s)
        setModels((m.ollama as string[]) || [])
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      }
    })()
  }, [])

  if (!settings) {
    return <p className="text-[var(--muted)]">Loading settings…</p>
  }

  const set = (key: string, value: string | number | boolean) => {
    setSettings({ ...settings, [key]: value })
    setSaved(false)
  }

  const save = async () => {
    try {
      const next = await api.updateSettings(settings)
      setSettings(next)
      setSaved(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <section className="rounded-2xl border border-[var(--line)] bg-[var(--bg1)]/80 p-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h2 className="text-lg font-semibold">Settings</h2>
        <button className="rounded-xl bg-[var(--accent)] px-4 py-2 text-black" onClick={() => void save()}>
          Save
        </button>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Field label="Performance profile">
          <select
            className="field"
            value={String(settings.performance_profile)}
            onChange={(e) => set('performance_profile', e.target.value)}
          >
            <option value="auto">auto</option>
            <option value="low">low</option>
            <option value="balanced">balanced</option>
            <option value="quality">quality</option>
          </select>
        </Field>

        <Field label="LLM model">
          <select
            className="field"
            value={String(settings.ollama_model)}
            onChange={(e) => set('ollama_model', e.target.value)}
          >
            {[String(settings.ollama_model), ...models.filter((m) => m !== settings.ollama_model)].map(
              (m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ),
            )}
            <option value="qwen3:1.7b">qwen3:1.7b</option>
            <option value="qwen3:4b-instruct">qwen3:4b-instruct</option>
          </select>
        </Field>

        <Field label="STT provider">
          <select
            className="field"
            value={String(settings.stt_provider)}
            onChange={(e) => set('stt_provider', e.target.value)}
          >
            <option value="moonshine">moonshine</option>
            <option value="whisper">whisper</option>
          </select>
        </Field>

        <Field label="TTS voice">
          <select
            className="field"
            value={String(settings.tts_voice)}
            onChange={(e) => set('tts_voice', e.target.value)}
          >
            {VOICES.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Speaking rate">
          <input
            className="field"
            type="number"
            step="0.05"
            min="0.5"
            max="2"
            value={Number(settings.tts_speed)}
            onChange={(e) => set('tts_speed', Number(e.target.value))}
          />
        </Field>

        <Field label="VAD confidence">
          <input
            className="field"
            type="number"
            step="0.05"
            min="0.1"
            max="1"
            value={Number(settings.vad_confidence)}
            onChange={(e) => set('vad_confidence', Number(e.target.value))}
          />
        </Field>

        <Field label="VAD stop seconds">
          <input
            className="field"
            type="number"
            step="0.05"
            min="0.05"
            max="2"
            value={Number(settings.vad_stop_secs)}
            onChange={(e) => set('vad_stop_secs', Number(e.target.value))}
          />
        </Field>

        <Field label="LLM temperature">
          <input
            className="field"
            type="number"
            step="0.05"
            min="0"
            max="1.5"
            value={Number(settings.llm_temperature)}
            onChange={(e) => set('llm_temperature', Number(e.target.value))}
          />
        </Field>

        <Field label="Context tokens">
          <input
            className="field"
            type="number"
            step="256"
            min="512"
            max="32768"
            value={Number(settings.llm_context_tokens)}
            onChange={(e) => set('llm_context_tokens', Number(e.target.value))}
          />
        </Field>

        <Field label="Push to talk fallback">
          <input
            type="checkbox"
            checked={Boolean(settings.push_to_talk)}
            onChange={(e) => set('push_to_talk', e.target.checked)}
          />
        </Field>

        <Field label="Thinking mode">
          <input
            type="checkbox"
            checked={Boolean(settings.thinking_mode)}
            onChange={(e) => set('thinking_mode', e.target.checked)}
          />
        </Field>
      </div>

      <Field label="System prompt">
        <textarea
          className="field min-h-28"
          value={String(settings.system_prompt || '')}
          onChange={(e) => set('system_prompt', e.target.value)}
        />
      </Field>

      {saved ? <p className="mt-3 text-sm text-[var(--accent)]">Saved.</p> : null}
      {error ? <p className="mt-3 text-sm text-[var(--danger)]">{error}</p> : null}

      <style>{`
        .field {
          width: 100%;
          border-radius: 0.75rem;
          border: 1px solid var(--line);
          background: var(--bg2);
          padding: 0.55rem 0.75rem;
          color: var(--text);
        }
      `}</style>
    </section>
  )
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block space-y-1 text-sm">
      <span className="text-[var(--muted)]">{label}</span>
      <div>{children}</div>
    </label>
  )
}
