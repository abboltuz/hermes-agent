import { describe, expect, it } from 'vitest'

import type { ClientSessionState } from '../../../types'

import {
  createPersistedDisplayTranscriptProvenance,
  hasPersistedDisplayTranscriptProvenance,
  invalidatePersistedDisplayTranscriptAuthority,
  shouldPaintPersistedTranscript,
  suppressTranscriptForView
} from './transcript-provenance'

const state = (messages = [{ id: 'm1' }] as never[]): ClientSessionState =>
  ({ messages, storedSessionId: 'stored-a' }) as unknown as ClientSessionState

describe('persisted transcript provenance', () => {
  it('requires the exact stored session and scope before publishing a warm tail', () => {
    const expected = createPersistedDisplayTranscriptProvenance({
      lineageRootId: 'root-a',
      scope: { connectionId: 'connection-a', profile: 'default' },
      storedSessionId: 'stored-a'
    })

    const proven = { ...state(), transcriptProvenance: expected }
    const wrongSession = { ...proven, transcriptProvenance: { ...expected, storedSessionId: 'stored-b' } }

    expect(hasPersistedDisplayTranscriptProvenance(proven, expected)).toBe(true)
    expect(hasPersistedDisplayTranscriptProvenance(wrongSession, expected)).toBe(false)
    expect(suppressTranscriptForView(state(), true).messages).toEqual([])
  })

  it('invalidates provenance when the persisted session id rotates', () => {
    const proven = {
      ...state(),
      transcriptProvenance: createPersistedDisplayTranscriptProvenance({
        lineageRootId: 'root-a',
        scope: 'default',
        storedSessionId: 'stored-a'
      })
    }

    const invalidated = invalidatePersistedDisplayTranscriptAuthority({ ...proven, storedSessionId: 'stored-b' })

    expect(invalidated.transcriptProvenance).toBeUndefined()
    expect(invalidated.transcriptAuthorityEpoch).toBe(1)
  })

  it('paints a resolved persisted snapshot only while the resume is current', () => {
    expect(shouldPaintPersistedTranscript(true, true)).toBe(true)
    expect(shouldPaintPersistedTranscript(true, false)).toBe(false)
    expect(shouldPaintPersistedTranscript(false, true)).toBe(false)
  })
})
