export type Session = {
  id: string
  title: string
  status: string
  summary?: string | null
  started_at?: string | null
  ended_at?: string | null
  updated_at?: string | null
  messages?: Message[]
}

export type Message = {
  id: string
  role: string
  content: string
  spoken_status?: string
  interrupted?: boolean
  created_at?: string | null
}

export type MemoryItem = {
  id: string
  type: string
  content: string
  importance: number
  confidence: number
  consent_status: string
  active: boolean
  created_at?: string | null
  updated_at?: string | null
}

export type TaskItem = {
  id: string
  title: string
  description: string
  status: string
}

export type NoteItem = {
  id: string
  title: string
  content: string
  tags: string[]
}

export type AppSettings = Record<string, string | number | boolean>

export type SystemStatus = {
  backend: string
  ollama: {
    ok: boolean
    base_url: string
    models: string[]
    error?: string | null
  }
  hardware: {
    os_name: string
    architecture: string
    cpu_model: string
    cpu_count: number
    ram_total_gb: number
    ram_available_gb: number
    gpu: {
      available: boolean
      name?: string | null
      vram_total_mb?: number | null
      cuda_available: boolean
    }
    recommended_profile: string
    missing_dependencies: string[]
    notes: string[]
  }
  active_profile: Record<string, unknown>
  metrics: Record<string, unknown>
}
