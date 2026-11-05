import { useEffect, useState } from 'react'
import { api } from '../services/api'
import type { NoteItem, TaskItem } from '../types'

export function ToolsPanel() {
  const [notes, setNotes] = useState<NoteItem[]>([])
  const [tasks, setTasks] = useState<TaskItem[]>([])
  const [noteTitle, setNoteTitle] = useState('')
  const [noteBody, setNoteBody] = useState('')
  const [taskTitle, setTaskTitle] = useState('')
  const [error, setError] = useState<string | null>(null)

  const refresh = async () => {
    const [n, t] = await Promise.all([api.listNotes(), api.listTasks()])
    setNotes(n.result?.notes || n.notes || [])
    setTasks(t.result?.tasks || t.tasks || [])
  }

  useEffect(() => {
    void refresh().catch((e) => setError(String(e)))
  }, [])

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <section className="rounded-2xl border border-[var(--line)] bg-[var(--bg1)]/80 p-4">
        <h2 className="mb-3 text-lg font-semibold">Notes</h2>
        <div className="mb-3 space-y-2">
          <input
            className="w-full rounded-xl border border-[var(--line)] bg-[var(--bg2)] px-3 py-2"
            placeholder="Title"
            value={noteTitle}
            onChange={(e) => setNoteTitle(e.target.value)}
          />
          <textarea
            className="w-full rounded-xl border border-[var(--line)] bg-[var(--bg2)] px-3 py-2"
            placeholder="Content"
            value={noteBody}
            onChange={(e) => setNoteBody(e.target.value)}
          />
          <button
            className="rounded-xl bg-[var(--accent)] px-4 py-2 text-black"
            onClick={() =>
              void api
                .createNote(noteTitle || 'Untitled', noteBody)
                .then(() => {
                  setNoteTitle('')
                  setNoteBody('')
                  return refresh()
                })
            }
          >
            Create note
          </button>
        </div>
        <div className="space-y-2">
          {notes.map((n) => (
            <div key={n.id} className="rounded-xl border border-[var(--line)] bg-[var(--bg2)] p-3 text-sm">
              <div className="font-medium">{n.title}</div>
              <div className="text-[var(--muted)]">{n.content}</div>
              <button
                className="mt-2 text-xs text-[var(--danger)]"
                onClick={() => void api.deleteNote(n.id).then(refresh)}
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      </section>

      <section className="rounded-2xl border border-[var(--line)] bg-[var(--bg1)]/80 p-4">
        <h2 className="mb-3 text-lg font-semibold">Tasks</h2>
        <div className="mb-3 flex gap-2">
          <input
            className="flex-1 rounded-xl border border-[var(--line)] bg-[var(--bg2)] px-3 py-2"
            placeholder="New task"
            value={taskTitle}
            onChange={(e) => setTaskTitle(e.target.value)}
          />
          <button
            className="rounded-xl bg-[var(--accent)] px-4 py-2 text-black"
            onClick={() =>
              void api.createTask(taskTitle).then(() => {
                setTaskTitle('')
                return refresh()
              })
            }
          >
            Add
          </button>
        </div>
        <div className="space-y-2">
          {tasks.map((t) => (
            <div key={t.id} className="flex items-center justify-between rounded-xl border border-[var(--line)] bg-[var(--bg2)] p-3 text-sm">
              <div>
                <div className={t.status === 'completed' ? 'line-through opacity-60' : ''}>{t.title}</div>
                <div className="text-xs text-[var(--muted)]">{t.status}</div>
              </div>
              <div className="flex gap-2">
                {t.status !== 'completed' ? (
                  <button
                    className="text-[var(--accent)]"
                    onClick={() => void api.completeTask(t.id).then(refresh)}
                  >
                    Done
                  </button>
                ) : null}
                <button
                  className="text-[var(--danger)]"
                  onClick={() => void api.deleteTask(t.id).then(refresh)}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
        {error ? <p className="mt-3 text-sm text-[var(--danger)]">{error}</p> : null}
      </section>
    </div>
  )
}
