import { describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { deferred } from '@/test/deferred'
import type { SessionResumeRetryResponse } from '@/types/hermes'

import { mergeSessionPreparation, retryRetainedSessionPreparation } from './retry-preparation'

describe('mergeSessionPreparation', () => {
  it('does not let a stale acknowledgement regress a terminal event', () => {
    const ready = { attempt: 1, message_count: 42, phase: 'history', status: 'ready' } as const
    const stale = { attempt: 1, phase: 'history', status: 'preparing' } as const

    expect(mergeSessionPreparation(ready, stale)).toBe(ready)
  })

  it('allows a newer explicit attempt to replace an older failure', () => {
    const failed = { attempt: 1, phase: 'history', status: 'preparation_failed' } as const
    const retrying = { attempt: 2, phase: 'history', status: 'preparing' } as const

    expect(mergeSessionPreparation(failed, retrying)).toBe(retrying)
  })
})

describe('retryRetainedSessionPreparation', () => {
  it('retries the retained runtime and changes only preparation state', async () => {
    const request = vi.fn(async () => ({
      session_id: 'runtime-1',
      preparation: { attempt: 2, phase: 'history', status: 'preparing' }
    }))

    let state = createClientSessionState('stored-1', [
      { id: 'durable-row', role: 'user', parts: [{ type: 'text', text: 'still readable' }] }
    ])

    const update = vi.fn((_runtimeId: string, updater: (current: typeof state) => typeof state) => {
      state = updater(state)
    })

    await retryRetainedSessionPreparation(request, update, 'runtime-1')

    expect(request).toHaveBeenCalledWith('session.resume.retry', { session_id: 'runtime-1' })
    expect(update).toHaveBeenCalledWith('runtime-1', expect.any(Function))
    expect(state.preparation).toEqual({ attempt: 2, phase: 'history', status: 'preparing' })
    expect(state.messages[0]?.id).toBe('durable-row')
  })

  it('does not regress a ready event when the retry response arrives later', async () => {
    const response = deferred<SessionResumeRetryResponse>()
    const request = vi.fn(() => response.promise)

    let state: ClientSessionState = {
      ...createClientSessionState('stored-1'),
      preparation: { attempt: 2, message_count: 42, phase: 'history', status: 'ready' } as const
    }

    const update = vi.fn((_runtimeId: string, updater: (current: typeof state) => typeof state) => {
      state = updater(state)
    })

    const retrying = retryRetainedSessionPreparation(request, update, 'runtime-1')
    response.resolve({
      session_id: 'runtime-1',
      preparation: { attempt: 2, phase: 'history', status: 'preparing' }
    })
    await retrying

    expect(state.preparation?.status).toBe('ready')
  })
})
