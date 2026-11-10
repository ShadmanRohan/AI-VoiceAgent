import { PipecatClient, RTVIEvent } from '@pipecat-ai/client-js'
import {
  PipecatClientAudio,
  PipecatClientProvider,
  usePipecatClient,
  usePipecatClientTransportState,
  useRTVIClientEvent,
} from '@pipecat-ai/client-react'
import {
  ProtobufFrameSerializer,
  WebSocketTransport,
} from '@pipecat-ai/websocket-transport'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../services/api'
import { AudioMeter } from './AudioMeter'

type ChatLine = {
  id: string
  role: 'user' | 'assistant' | 'system'
  text: string
  partial?: boolean
}

function VoiceInner({
  onSessionHint,
}: {
  onSessionHint: (id: string | null) => void
}) {
  const client = usePipecatClient()
  const transportState = usePipecatClientTransportState()
  const [lines, setLines] = useState<ChatLine[]>([])
  const [partialUser, setPartialUser] = useState('')
  const [partialBot, setPartialBot] = useState('')
  const [muted, setMuted] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [level, setLevel] = useState(0)
  const [aiSpeaking, setAiSpeaking] = useState(false)
  const [textInput, setTextInput] = useState('')
  const [textSessionId, setTextSessionId] = useState<string | null>(null)
  const [busyText, setBusyText] = useState(false)
  const [langInfo, setLangInfo] = useState<string>('auto')
  const botBuf = useRef('')

  const isConnected = transportState === 'ready'
  const isConnecting = ['authenticating', 'connecting', 'connected'].includes(transportState)

  useRTVIClientEvent(
    RTVIEvent.Error,
    useCallback((err: unknown) => {
      setError(typeof err === 'string' ? err : JSON.stringify(err))
    }, []),
  )

  useRTVIClientEvent(
    RTVIEvent.UserTranscript,
    useCallback((data: { text?: string; final?: boolean }) => {
      const text = data.text || ''
      if (!data.final) {
        setPartialUser(text)
        return
      }
      setPartialUser('')
      if (text.trim()) {
        setLines((prev) => [...prev, { id: crypto.randomUUID(), role: 'user', text }])
      }
    }, []),
  )

  useRTVIClientEvent(
    RTVIEvent.BotTranscript,
    useCallback((data: { text?: string }) => {
      const text = data.text || ''
      botBuf.current += text
      setPartialBot(botBuf.current)
    }, []),
  )

  useRTVIClientEvent(
    RTVIEvent.BotStoppedSpeaking,
    useCallback(() => {
      setAiSpeaking(false)
      // Re-enable mic after AI audio finishes (unless user muted manually).
      if (client && !muted) {
        void client.enableMic(true)
      }
      const text = botBuf.current.trim()
      botBuf.current = ''
      setPartialBot('')
      if (text) {
        setLines((prev) => [...prev, { id: crypto.randomUUID(), role: 'assistant', text }])
      }
    }, [client, muted]),
  )

  useRTVIClientEvent(
    RTVIEvent.BotStartedSpeaking,
    useCallback(() => {
      setAiSpeaking(true)
      // Mute mic while AI speaks so speaker echo cannot barge-in and cut TTS.
      if (client) {
        void client.enableMic(false)
      }
    }, [client]),
  )

  useEffect(() => {
    if (!isConnected || !client) return
    let raf = 0
    const tick = () => {
      try {
        // Approximate activity from mute + speaking flags
        setLevel(aiSpeaking ? 0.35 + Math.random() * 0.4 : muted ? 0 : 0.15 + Math.random() * 0.25)
      } catch {
        /* ignore */
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [aiSpeaking, client, isConnected, muted])

  const connect = async () => {
    if (!client) return
    setError(null)
    try {
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      const base = import.meta.env.VITE_API_BASE || ''
      const wsUrl = base
        ? `${base.replace(/^http/, 'ws')}/ws`
        : `${proto}://${window.location.host}/ws`
      await client.connect({ wsUrl })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const disconnect = async () => {
    if (!client) return
    await client.disconnect()
    setPartialUser('')
    setPartialBot('')
    setAiSpeaking(false)
  }

  const toggleMute = async () => {
    if (!client) return
    const next = !muted
    setMuted(next)
    await client.enableMic(!next)
  }

  const interrupt = async () => {
    try {
      await api.interrupt(textSessionId || undefined)
      if (client && isConnected) {
        // Local immediate UX feedback; server pipeline also handles barge-in via VAD
        setAiSpeaking(false)
        botBuf.current = ''
        setPartialBot('')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const clearChat = () => {
    setLines([])
    setPartialUser('')
    setPartialBot('')
    botBuf.current = ''
    setTextSessionId(null)
    onSessionHint(null)
  }

  const sendText = async () => {
    const text = textInput.trim()
    if (!text || busyText) return
    setBusyText(true)
    setError(null)
    setLines((prev) => [...prev, { id: crypto.randomUUID(), role: 'user', text }])
    setTextInput('')
    try {
      const res = await api.chat(text, textSessionId || undefined)
      setTextSessionId(res.session_id)
      onSessionHint(res.session_id)
      const langEvent = (res as { events?: Array<Record<string, string>> }).events?.find(
        (e) => e.type === 'language',
      )
      if (langEvent) {
        setLangInfo(
          `${langEvent.detected || '?'} → ${langEvent.response_language || '?'} (${langEvent.tts_provider || ''})`,
        )
      }
      setLines((prev) => [
        ...prev,
        { id: res.message_id || crypto.randomUUID(), role: 'assistant', text: res.text },
      ])
      await api.ackPlayback(res.message_id, 'spoken').catch(() => undefined)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusyText(false)
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[1.4fr_0.9fr]">
      <section className="rounded-2xl border border-[var(--line)] bg-[var(--bg1)]/80 p-5 backdrop-blur">
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <button
            className="rounded-xl bg-[var(--accent)] px-4 py-2 font-medium text-black disabled:opacity-50"
            onClick={isConnected ? disconnect : connect}
            disabled={isConnecting}
          >
            {isConnected ? 'Stop conversation' : isConnecting ? 'Connecting…' : 'Start voice'}
          </button>
          <button
            className="rounded-xl border border-[var(--line)] px-4 py-2 disabled:opacity-40"
            onClick={toggleMute}
            disabled={!isConnected}
          >
            {muted ? 'Unmute mic' : 'Mute mic'}
          </button>
          <button
            className="rounded-xl border border-[var(--danger)] px-4 py-2 text-[var(--danger)] disabled:opacity-40"
            onClick={interrupt}
            disabled={!aiSpeaking && !partialBot}
          >
            Interrupt AI
          </button>
          <button className="rounded-xl border border-[var(--line)] px-4 py-2" onClick={clearChat}>
            Clear
          </button>
          <span className="text-sm text-[var(--muted)]">
            Transport: <strong className="text-[var(--text)]">{transportState}</strong>
          </span>
        </div>

        <div className="mb-4 grid gap-3 sm:grid-cols-2">
          <AudioMeter
            level={isConnected && !muted ? level : 0}
            active={isConnected && !muted}
            label="Microphone"
          />
          <AudioMeter level={aiSpeaking ? level : 0} active={aiSpeaking} label="AI speaking" />
        </div>

        <div className="mb-3 flex flex-wrap gap-3 text-sm">
          <StatusChip on={isConnected} label="Connected" />
          <StatusChip on={isConnected && !muted} label="Mic live" />
          <StatusChip on={aiSpeaking} label="AI audio" />
          <StatusChip on={true} label={`Lang ${langInfo}`} />
        </div>

        <div className="h-[420px] space-y-3 overflow-y-auto rounded-xl border border-[var(--line)] bg-black/20 p-4">
          {lines.length === 0 && !partialUser && !partialBot ? (
            <p className="text-[var(--muted)]">
              Start a voice session, or type below. Continuous listening is enabled while connected —
              no push-to-talk required unless configured.
            </p>
          ) : null}
          {lines.map((line) => (
            <div key={line.id} className="text-sm leading-relaxed">
              <span
                className="mr-2 font-semibold"
                style={{ color: line.role === 'user' ? 'var(--user)' : 'var(--bot)' }}
              >
                {line.role === 'user' ? 'You' : 'AI'}
              </span>
              {line.text}
            </div>
          ))}
          {partialUser ? (
            <div className="text-sm italic text-[var(--muted)]">
              <span className="mr-2 font-semibold text-[var(--user)]">You</span>
              {partialUser}
            </div>
          ) : null}
          {partialBot ? (
            <div className="text-sm italic text-[var(--muted)]">
              <span className="mr-2 font-semibold text-[var(--bot)]">AI</span>
              {partialBot}
            </div>
          ) : null}
        </div>

        <div className="mt-4 flex gap-2">
          <input
            className="flex-1 rounded-xl border border-[var(--line)] bg-[var(--bg2)] px-3 py-2 outline-none focus:border-[var(--accent)]"
            placeholder="Type a message (text fallback / tests)"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void sendText()
            }}
          />
          <button
            className="rounded-xl bg-[var(--accent-dim)] px-4 py-2 text-white disabled:opacity-50"
            onClick={() => void sendText()}
            disabled={busyText}
          >
            Send
          </button>
        </div>
        {error ? <p className="mt-3 text-sm text-[var(--danger)]">{error}</p> : null}
      </section>

      <aside className="rounded-2xl border border-[var(--line)] bg-[var(--bg1)]/80 p-5">
        <h2 className="mb-2 text-lg font-semibold">Live session</h2>
        <ul className="space-y-2 text-sm text-[var(--muted)]">
          <li>Partial transcripts appear italic until finalized.</li>
          <li>Only finalized user turns are stored in history.</li>
          <li>Mic mutes while the AI speaks so echo cannot cut the reply; use Interrupt to stop.</li>
          <li>Text send path exercises memory + Ollama without mic.</li>
        </ul>
        {textSessionId ? (
          <p className="mt-4 text-xs text-[var(--muted)]">Text session: {textSessionId}</p>
        ) : null}
      </aside>
    </div>
  )
}

function StatusChip({ on, label }: { on: boolean; label: string }) {
  return (
    <span
      className="inline-flex items-center gap-2 rounded-full border border-[var(--line)] px-3 py-1"
      style={{ color: on ? 'var(--accent)' : 'var(--muted)' }}
    >
      <span
        className="h-2 w-2 rounded-full"
        style={{ background: on ? 'var(--accent)' : '#44554c' }}
      />
      {label}
    </span>
  )
}

export function VoicePanel({ onSessionHint }: { onSessionHint: (id: string | null) => void }) {
  const client = useMemo(
    () =>
      new PipecatClient({
        // WebSocket audio works through the Cloudflare tunnel (WebRTC/TURN fails on CGNAT).
        transport: new WebSocketTransport({
          serializer: new ProtobufFrameSerializer(),
          recorderSampleRate: 16000,
          playerSampleRate: 24000,
        }),
        enableMic: true,
      }),
    [],
  )

  return (
    <PipecatClientProvider client={client}>
      <VoiceInner onSessionHint={onSessionHint} />
      <PipecatClientAudio />
    </PipecatClientProvider>
  )
}
