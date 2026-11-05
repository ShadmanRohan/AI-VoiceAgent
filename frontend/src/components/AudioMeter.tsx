type Props = {
  level: number
  active?: boolean
  label?: string
}

export function AudioMeter({ level, active = false, label }: Props) {
  const bars = 16
  return (
    <div className="flex flex-col gap-2">
      {label ? <div className="text-xs uppercase tracking-wide text-[var(--muted)]">{label}</div> : null}
      <div className="flex h-10 items-end gap-1 rounded-lg border border-[var(--line)] bg-[var(--bg2)] px-2 py-1">
        {Array.from({ length: bars }).map((_, i) => {
          const threshold = (i + 1) / bars
          const on = active && level >= threshold * 0.85
          return (
            <div
              key={i}
              className="w-full rounded-sm transition-all duration-75"
              style={{
                height: `${18 + i * 4}%`,
                background: on ? 'var(--accent)' : '#24332c',
                opacity: on ? 0.5 + threshold * 0.5 : 0.55,
              }}
            />
          )
        })}
      </div>
    </div>
  )
}
