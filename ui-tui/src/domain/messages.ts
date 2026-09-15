import { LONG_MSG } from '../config/limits.js'
import { buildToolTrailLine } from '../lib/text.js'
import type { Msg, SessionInfo } from '../types.js'

export const introMsg = (info: SessionInfo): Msg => ({ info, kind: 'intro', role: 'system', text: '' })

export const userDisplay = (text: string) => {
  if (text.length <= LONG_MSG) {
    return text
  }

  const first = text.split('\n')[0]?.trim() ?? ''
  const words = first.split(/\s+/).filter(Boolean)
  const prefix = (words.length > 1 ? words.slice(0, 4).join(' ') : first).slice(0, 80)

  return `${prefix || '(message)'} [long message]`
}

export const toTranscriptMessages = (rows: unknown): Msg[] => {
  if (!Array.isArray(rows)) {
    return []
  }

  const out: Msg[] = []
  let pending: string[] = []

  for (const row of rows) {
    if (!row || typeof row !== 'object') {
      continue
    }

    const {
      context,
      display_kind,
      name,
      origin_kind,
      provenance_metadata,
      role,
      text,
      timestamp,
      trust_kind,
      turn_kind
    } = row as TranscriptRow

    const createdAt =
      typeof timestamp === 'number' && Number.isFinite(timestamp) && timestamp > 0 ? timestamp : undefined

    if (role === 'tool') {
      pending.push(buildToolTrailLine(name ?? 'tool', context ?? ''))

      continue
    }

    if (typeof text !== 'string' || !text.trim()) {
      continue
    }

    // Display-only timeline events: render as dim ◈ markers instead of
    // opaque user messages. Hidden compaction handoffs are skipped entirely.
    if (display_kind === 'hidden' || turn_kind === 'runtime_scaffolding') {
      continue
    }

    if (display_kind === 'model_switch') {
      out.push({ kind: 'event', role: 'system', text: 'model changed' })
      pending = []

      continue
    }

    if (display_kind === 'auto_continue') {
      out.push({ kind: 'event', role: 'system', text: 'resumed interrupted turn' })
      pending = []

      continue
    }

    if (display_kind === 'personality_switch') {
      out.push({ kind: 'event', role: 'system', text: 'personality changed' })
      pending = []

      continue
    }

    if (display_kind === 'async_delegation_complete') {
      const meta = (row as TranscriptRow).display_metadata
      const count = meta && typeof meta.task_count === 'number' ? meta.task_count : undefined

      const label =
        count === undefined
          ? 'background agent work finished'
          : `${count} background agent${count === 1 ? '' : 's'} finished`

      out.push({ kind: 'event', role: 'system', text: label })
      pending = []

      continue
    }

    if (role === 'assistant') {
      out.push({ role, text, ...(createdAt !== undefined && { createdAt }), ...(pending.length && { tools: pending }) })
      pending = []
    } else if (role === 'user' || role === 'system') {
      const isLocalUserActor =
        origin_kind === 'human_user' &&
        (turn_kind === 'prompt' || turn_kind === 'task_instruction' || turn_kind === 'ui_action') &&
        trust_kind === 'user_authorized'

      const displayRole = role === 'user' && !isLocalUserActor ? 'system' : role

      const actorLabel =
        !origin_kind || origin_kind === 'legacy_unknown'
          ? 'Source unknown'
          : origin_kind === 'external_actor'
            ? 'External actor'
            : String(origin_kind || 'system').replaceAll('_', ' ')

      const displayText = displayRole === 'system' && role === 'user' ? `[${actorLabel}] ${text}` : text

      const messageId =
        typeof provenance_metadata?.message_id === 'string' && provenance_metadata.message_id.trim()
          ? provenance_metadata.message_id.trim()
          : undefined

      out.push({
        role: displayRole,
        text: displayText,
        ...(createdAt !== undefined && { createdAt }),
        ...(messageId && { messageId }),
        ...(origin_kind && { originKind: origin_kind }),
        ...(turn_kind && { turnKind: turn_kind }),
        ...(trust_kind && { trustKind: trust_kind })
      })
      pending = []
    }
  }

  return out
}

export const fmtDuration = (ms: number) => {
  const t = Math.max(0, Math.floor(ms / 1000))
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  const s = t % 60

  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`
}

interface TranscriptRow {
  context?: string
  display_kind?: string
  display_metadata?: { task_count?: number; [key: string]: unknown }
  name?: string
  origin_kind?: string
  provenance_metadata?: Record<string, boolean | number | string>
  role?: string
  text?: string
  timestamp?: number
  trust_kind?: string
  turn_kind?: string
}
