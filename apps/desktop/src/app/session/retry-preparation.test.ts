import { describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'

import { retryRetainedSessionPreparation } from './retry-preparation'

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
})
