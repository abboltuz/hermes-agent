import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $notifications, clearNotifications } from '@/store/notifications'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'retained-runtime'

let stream: MessageStreamHarness

describe('deferred resume preparation events', () => {
  beforeEach(() => {
    clearNotifications()
    stream = renderMessageStream(SID)
    stream.states.set(
      SID,
      createClientSessionState('stored-large-chat', [
        { id: 'durable-user', role: 'user', parts: [textPart('readable archive row')] }
      ])
    )
  })

  afterEach(() => {
    cleanup()
    clearNotifications()
  })

  it('tracks preparation independently of the readable transcript', () => {
    act(() =>
      stream.handleEvent({
        payload: {
          preparation: { attempt: 1, phase: 'history', status: 'preparing' },
          status: 'loading'
        },
        session_id: SID,
        type: 'session.resume_progress'
      })
    )

    expect(stream.state().preparation).toEqual({ attempt: 1, phase: 'history', status: 'preparing' })
    expect(stream.state().messages[0]?.id).toBe('durable-user')

    act(() =>
      stream.handleEvent({
        payload: {
          message: 'resume failed: history is too large',
          preparation: {
            attempt: 1,
            message: 'resume failed: history is too large',
            phase: 'history',
            status: 'preparation_failed'
          },
          status: 'failed'
        },
        session_id: SID,
        type: 'session.resume_progress'
      })
    )

    expect(stream.state().preparation?.status).toBe('preparation_failed')
    expect(stream.state().messages).toHaveLength(1)
  })

  it('does not turn the compatibility error into a failed assistant turn', () => {
    act(() =>
      stream.handleEvent({
        payload: { kind: 'session_preparation', message: 'resume failed: history is too large' },
        session_id: SID,
        type: 'error'
      })
    )

    expect(stream.state().messages).toHaveLength(1)
    expect(stream.state().messages.some(message => Boolean(message.error))).toBe(false)
    expect($notifications.get()).toHaveLength(0)
  })

  it('publishes the ready state after an explicit retry succeeds', () => {
    act(() =>
      stream.handleEvent({
        payload: {
          preparation: { attempt: 2, message_count: 42, phase: 'history', status: 'ready' },
          status: 'complete'
        },
        session_id: SID,
        type: 'session.resume_progress'
      })
    )

    expect(stream.state().preparation).toEqual({
      attempt: 2,
      message_count: 42,
      phase: 'history',
      status: 'ready'
    })
  })

  it('ignores a stale preparing acknowledgement received after ready', () => {
    act(() =>
      stream.handleEvent({
        payload: {
          preparation: { attempt: 1, message_count: 42, phase: 'history', status: 'ready' }
        },
        session_id: SID,
        type: 'session.resume_progress'
      })
    )
    act(() =>
      stream.handleEvent({
        payload: {
          preparation: { attempt: 1, phase: 'history', status: 'preparing' }
        },
        session_id: SID,
        type: 'session.resume_progress'
      })
    )

    expect(stream.state().preparation?.status).toBe('ready')
  })
})
