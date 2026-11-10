import { useEffect, useState, type ReactNode } from 'react'
import { api } from '../services/api'
import type { AppSettings } from '../types'

const EN_VOICES = [
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

const BN_VOICES = ['Aditi', 'Arjun', 'Tapan', 'Rashmi']

export function SettingsPanel() {
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [models, setModels] = useState<string[]>([])
  const [readiness, setReadiness] = useState<Record<string, unknown> | null>(null)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    void (async () => {
      try {
        const [s, m] = await Promise.all([api.getSettings(), api.models()])
        setSettings(s)
        setModels((m.ollama as string[]) || [])
        setReadiness((m.readiness as Record<string, unknown>) || null)
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
      const m = await api.models()
      setReadiness((m.readiness as Record<string, unknown>) || null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const bnTts = (readiness?.models as Record<string, any> | undefined)?.bengali_tts

  return (
    <section className="rounded-2xl border border-[var(--line)] bg-[var(--bg1)]/80 p-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h2 className="text-lg font-semibold">Settings</h2>
        <button className="rounded-xl bg-[var(--accent)] px-4 py-2 text-black" onClick={() => void save()}>
          Save
        </button>
      </div>

      <div className="mb-5 rounded-xl border border-[var(--line)] bg-[var(--bg2)] p-3 text-sm">
        <div className="mb-1 font-medium">Model readiness</div>
        <ul className="space-y-1 text-[var(--muted)]">
          <li>Language mode: {String(settings.language_mode || 'auto')}</li>
          <li>LLM: {String(settings.ollama_model)}</li>
          <li>STT: {String(settings.stt_provider)} / {String(settings.stt_model || '')}</li>
          <li>
            Bengali TTS:{' '}
            {bnTts?.available
              ? `available (${bnTts.device || 'device TBD'})`
              : bnTts?.error || 'unavailable — set HF_TOKEN after accepting model terms'}
          </li>
          <li>English TTS: Kokoro ({String(settings.tts_voice)})</li>
          <li>HF token configured: {String(settings.hf_token_configured)}</li>
        </ul>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Field label="Language mode">
          <select
            className="field"
            value={String(settings.language_mode || 'auto')}
            onChange={(e) => set('language_mode', e.target.value)}
          >
            <option value="auto">Auto</option>
            <option value="bangla">বাংলা</option>
            <option value="english">English</option>
          </select>
        </Field>

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
            {['qwen3.5:4b', 'qwen3:1.7b', 'qwen3:4b-instruct', ...models]
              .filter((v, i, a) => a.indexOf(v) === i)
              .map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
          </select>
        </Field>

        <Field label="Fallback LLM">
          <select
            className="field"
            value={String(settings.ollama_model_fallback || 'qwen3:1.7b')}
            onChange={(e) => set('ollama_model_fallback', e.target.value)}
          >
            <option value="qwen3:1.7b">qwen3:1.7b</option>
            <option value="qwen3.5:4b">qwen3.5:4b</option>
          </select>
        </Field>

        <Field label="STT provider">
          <select
            className="field"
            value={String(settings.stt_provider)}
            onChange={(e) => set('stt_provider', e.target.value)}
          >
            <option value="whisper">Whisper large-v3 (best Bengali)</option>
            <option value="whisper_turbo">Whisper Turbo (faster, weaker BN)</option>
            <option value="moonshine">Moonshine (fallback)</option>
          </select>
        </Field>

        <Field label="STT model">
          <select
            className="field"
            value={String(settings.stt_model || 'large-v3')}
            onChange={(e) => set('stt_model', e.target.value)}
          >
            <option value="large-v3">large-v3</option>
            <option value="large-v3-turbo">large-v3-turbo</option>
            <option value="small">small</option>
            <option value="base">base</option>
            <option value="tiny">tiny</option>
          </select>
        </Field>

        <Field label="Bengali TTS voice (Indic Parler)">
          <select
            className="field"
            value={String(settings.bengali_tts_voice || 'Aditi')}
            onChange={(e) => set('bengali_tts_voice', e.target.value)}
          >
            {BN_VOICES.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </Field>

        <Field label="English TTS voice (Kokoro)">
          <select
            className="field"
            value={String(settings.tts_voice)}
            onChange={(e) => set('tts_voice', e.target.value)}
          >
            {EN_VOICES.map((v) => (
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
