import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { turnController } from '../app/turnController.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import {
  hydrateLiveSessionInflight,
  liveSessionInflightMessages,
  scheduleResumeScrollToBottom,
  signalFreshSessionBoundary,
  writeActiveSessionFile
} from '../app/useSessionLifecycle.js'

describe('fresh session boundary', () => {
  it('signals only when a live session is replaced by a different session', () => {
    const onFreshSessionStarted = vi.fn()

    expect(signalFreshSessionBoundary('old-session', 'new-session', onFreshSessionStarted)).toBe(true)
    expect(signalFreshSessionBoundary(null, 'first-session', onFreshSessionStarted)).toBe(false)
    expect(signalFreshSessionBoundary('same-session', 'same-session', onFreshSessionStarted)).toBe(false)
    expect(signalFreshSessionBoundary('old-session', null, onFreshSessionStarted)).toBe(false)
    expect(signalFreshSessionBoundary('old-session', 'new-session')).toBe(false)
    expect(onFreshSessionStarted).toHaveBeenCalledOnce()
    expect(onFreshSessionStarted).toHaveBeenCalledWith('new-session')
  })
})

describe('writeActiveSessionFile', () => {
  let dir = ''

  afterEach(() => {
    if (dir) {
      rmSync(dir, { force: true, recursive: true })
      dir = ''
    }
  })

  it('writes the actual resumed session id for the shell exit summary', () => {
    dir = mkdtempSync(join(tmpdir(), 'hermes-tui-active-'))
    const path = join(dir, 'active.json')

    writeActiveSessionFile('actual_session', path)

    expect(JSON.parse(readFileSync(path, 'utf8'))).toEqual({ session_id: 'actual_session' })
  })
})

describe('live session activation in-flight state', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    turnController.fullReset()
    patchUiState({ streaming: true })
  })

  it('keeps the in-flight user prompt in history and hydrates partial assistant text', () => {
    const inflight = {
      assistant: 'partial answer',
      streaming: true,
      user: 'write a long answer',
      origin_kind: 'human_user',
      turn_kind: 'prompt',
      trust_kind: 'user_authorized',
      provenance_metadata: { message_id: 'tui:prompt-1' }
    }

    expect(liveSessionInflightMessages(inflight)).toEqual([
      {
        role: 'user',
        text: 'write a long answer',
        messageId: 'tui:prompt-1',
        originKind: 'human_user',
        turnKind: 'prompt',
        trustKind: 'user_authorized'
      }
    ])

    hydrateLiveSessionInflight(inflight)

    expect(turnController.bufRef).toBe('partial answer')
    expect(getTurnState().streaming).toBe('partial answer')
  })

  it('fails closed when an older gateway omits transient provenance', () => {
    expect(liveSessionInflightMessages({ assistant: '', streaming: true, user: 'ambiguous text' })).toEqual([
      { role: 'system', text: '[Source unknown] ambiguous text' }
    ])
  })

  it('projects an active internal wake as a system actor with semantic identity', () => {
    const inflight = {
      assistant: '',
      streaming: true,
      user: 'internal model payload',
      display_kind: 'internal_notification',
      display_metadata: { display_text: 'Task finished' },
      origin_kind: 'agent',
      turn_kind: 'continuation',
      trust_kind: 'trusted_internal',
      provenance_metadata: { message_id: 'kanban-wake:route:event-17' }
    }

    expect(liveSessionInflightMessages(inflight)).toEqual([
      {
        role: 'system',
        text: '[agent] Task finished',
        messageId: 'kanban-wake:route:event-17',
        originKind: 'agent',
        turnKind: 'continuation',
        trustKind: 'trusted_internal'
      }
    ])
  })

  it('composes durable, in-flight, and queued copies by immutable identity', () => {
    const semantic = {
      origin_kind: 'agent',
      turn_kind: 'continuation',
      trust_kind: 'trusted_internal',
      provenance_metadata: { message_id: 'kanban-wake:route:event-17' }
    }

    const transcript = [
      {
        role: 'system' as const,
        text: '[agent] Task finished',
        messageId: 'kanban-wake:route:event-17'
      }
    ]

    expect(
      liveSessionInflightMessages(
        { assistant: '', streaming: true, user: 'internal payload', ...semantic },
        { user: 'same queued payload', ...semantic },
        transcript
      )
    ).toEqual([])
  })

  it('restores the complete queued FIFO and keeps repeated text identities distinct', () => {
    const queued = (messageId: string) => ({
      user: 'repeat this',
      origin_kind: 'human_user',
      turn_kind: 'prompt',
      trust_kind: 'user_authorized',
      provenance_metadata: { message_id: messageId }
    })

    expect(liveSessionInflightMessages(null, null, [], [queued('tui:q1'), queued('tui:q2')])).toEqual([
      expect.objectContaining({ role: 'user', text: 'repeat this', messageId: 'tui:q1' }),
      expect.objectContaining({ role: 'user', text: 'repeat this', messageId: 'tui:q2' })
    ])
  })

  it('ignores empty in-flight payloads', () => {
    expect(liveSessionInflightMessages({ assistant: '', streaming: false, user: '   ' })).toEqual([])

    hydrateLiveSessionInflight({ assistant: '', streaming: false, user: '' })

    expect(turnController.bufRef).toBe('')
    expect(getTurnState().streaming).toBe('')
  })
})

describe('resume scroll settle', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('re-snaps while sticky and stops when the user scrolls away', () => {
    vi.useFakeTimers()
    let sticky = true
    let lastManualScrollAt = 0
    const scrollToBottom = vi.fn()

    const cancel = scheduleResumeScrollToBottom(
      {
        current: {
          getLastManualScrollAt: () => lastManualScrollAt,
          isSticky: () => sticky,
          scrollToBottom
        }
      } as any,
      [0, 80, 240]
    )

    vi.advanceTimersByTime(0)
    expect(scrollToBottom).toHaveBeenCalledTimes(1)

    vi.advanceTimersByTime(80)
    expect(scrollToBottom).toHaveBeenCalledTimes(2)

    sticky = false
    lastManualScrollAt = Date.now() + 1
    vi.advanceTimersByTime(160)
    expect(scrollToBottom).toHaveBeenCalledTimes(2)

    cancel()
  })

  it('cancels pending resume snaps', () => {
    vi.useFakeTimers()
    const scrollToBottom = vi.fn()

    const cancel = scheduleResumeScrollToBottom(
      {
        current: {
          getLastManualScrollAt: () => 0,
          isSticky: () => true,
          scrollToBottom
        }
      } as any,
      [20]
    )

    cancel()
    vi.advanceTimersByTime(20)

    expect(scrollToBottom).not.toHaveBeenCalled()
  })

  it('keeps the immediate resume snap even before sticky state settles', () => {
    vi.useFakeTimers()
    let sticky = false
    const scrollToBottom = vi.fn()

    const cancel = scheduleResumeScrollToBottom(
      {
        current: {
          getLastManualScrollAt: () => 0,
          isSticky: () => sticky,
          scrollToBottom
        }
      } as any,
      [0, 80]
    )

    vi.advanceTimersByTime(0)
    expect(scrollToBottom).toHaveBeenCalledTimes(1)

    vi.advanceTimersByTime(80)
    expect(scrollToBottom).toHaveBeenCalledTimes(1)

    sticky = true
    cancel()
  })
})
