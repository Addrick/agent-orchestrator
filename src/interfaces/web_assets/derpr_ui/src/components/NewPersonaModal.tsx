import { useState } from 'react'
import type { PortalStore } from '../state/store'

// Create-persona modal — a trimmed clone of the Inspector's persona pane. It
// captures the identity + base config a new persona needs to exist; tools,
// service bindings, and the kobold samplers are left to the full Inspector once
// the persona is created and switched to (POST /personas seeds defaults the
// existing PATCH/dev_command surface then edits). Name is the only required
// field; blank prompt/model fall back to the engine's defaults.
const MEMORY_MODES = [
  'CHANNEL_ISOLATED',
  'SERVER_WIDE',
  'PERSONAL',
  'GLOBAL',
  'TICKET_ISOLATED',
]

// Name is the routing key + a single dev-command token → must match the
// engine's server-side rule (lowercase [a-z0-9_-], no spaces). Validated here
// for instant feedback; the engine re-validates authoritatively.
const NAME_RE = /^[a-z0-9_-]+$/

interface Props {
  store: PortalStore
  onClose: () => void
}

export function NewPersonaModal({ store, onClose }: Props) {
  const { modelList, createPersona } = store
  const [name, setName] = useState('')
  const [prompt, setPrompt] = useState('')
  const [modelName, setModelName] = useState('')
  const [memoryMode, setMemoryMode] = useState('')
  const [temperature, setTemperature] = useState('')
  const [maxTokens, setMaxTokens] = useState('')
  const [historyMessages, setHistoryMessages] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const normName = name.trim().toLowerCase()
  const nameValid = normName !== '' && NAME_RE.test(normName)

  const onCreate = async () => {
    if (!nameValid || busy) return
    const body: Record<string, unknown> = { name: normName }
    if (prompt.trim()) body.prompt = prompt
    if (modelName) body.model_name = modelName
    if (memoryMode) body.memory_mode = memoryMode
    if (temperature.trim() !== '') body.temperature = Number(temperature)
    if (maxTokens.trim() !== '') body.max_tokens = Number(maxTokens)
    if (historyMessages.trim() !== '') body.history_messages = Number(historyMessages)

    setBusy(true)
    setError(null)
    const res = await createPersona(body)
    setBusy(false)
    if (res.ok) onClose()
    else setError(res.error || 'create failed')
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label="create persona"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <b>New persona</b>
          <button className="cbtn" title="close" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="modal-body">
          <div className="field">
            <span className="lbl">name · routing key</span>
            <div className="ctrl">
              <input
                autoFocus
                value={name}
                placeholder="e.g. researcher"
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') onCreate()
                }}
              />
            </div>
            {name.trim() !== '' && !nameValid && (
              <span className="savestatus err" style={{ marginTop: 6, display: 'block' }}>
                lowercase letters, digits, '-' and '_' only (no spaces)
              </span>
            )}
          </div>

          <div className="field">
            <span className="lbl">system prompt</span>
            <textarea
              className="ctrl area edit"
              value={prompt}
              placeholder={`(optional) defaults to "you are in character as ${normName || 'name'}"`}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </div>

          <div className="field">
            <div className="row2">
              <div>
                <span className="lbl">model_name</span>
                <div className="ctrl">
                  <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
                    <option value="">default</option>
                    {modelList.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div>
                <span className="lbl">memory_mode</span>
                <div className="ctrl">
                  <select value={memoryMode} onChange={(e) => setMemoryMode(e.target.value)}>
                    <option value="">default</option>
                    {MEMORY_MODES.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          </div>

          <div className="field">
            <div className="row2">
              <div>
                <span className="lbl">temperature</span>
                <div className="ctrl">
                  <input
                    type="number"
                    step="0.05"
                    value={temperature}
                    placeholder="default"
                    onChange={(e) => setTemperature(e.target.value)}
                  />
                </div>
              </div>
              <div>
                <span className="lbl">max_tokens</span>
                <div className="ctrl">
                  <input
                    type="number"
                    value={maxTokens}
                    placeholder="default"
                    onChange={(e) => setMaxTokens(e.target.value)}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="field">
            <span className="lbl">history_messages</span>
            <div className="ctrl">
              <input
                type="number"
                value={historyMessages}
                placeholder="default"
                onChange={(e) => setHistoryMessages(e.target.value)}
              />
            </div>
            <span className="modal-hint">
              tools, service bindings, and samplers are configured in the Inspector after creation
            </span>
          </div>
        </div>

        <div className="savebar">
          {error && <span className="savestatus err">{error}</span>}
          <span className="grow" />
          <button className="mini" onClick={onClose} disabled={busy}>
            cancel
          </button>
          <button className="savebtn" onClick={onCreate} disabled={!nameValid || busy}>
            {busy ? 'creating…' : 'create'}
          </button>
        </div>
      </div>
    </div>
  )
}
