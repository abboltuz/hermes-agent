import type { ClientSessionState, PersistedDisplayTranscriptProvenance } from '../../../types'

export type TranscriptProvenanceScope = string | null | undefined | { connectionId?: string | null; profile?: string | null }

export function createPersistedDisplayTranscriptProvenance({
  lineageRootId,
  scope,
  storedSessionId
}: {
  storedSessionId: string
  lineageRootId: string | null
  scope: TranscriptProvenanceScope
}): PersistedDisplayTranscriptProvenance {
  const connectionId = typeof scope === 'object' && scope ? (scope.connectionId ?? '').trim() : ''
  const profile = (typeof scope === 'string' ? scope : scope?.profile)?.trim() || 'default'
  return { connectionId, coverage: 'latest-page', lineageRootId, profile, source: 'persisted-display', storedSessionId }
}

export function hasPersistedDisplayTranscriptProvenance(
  state: Pick<ClientSessionState, 'transcriptProvenance'>,
  expected: PersistedDisplayTranscriptProvenance
): boolean {
  const actual = state.transcriptProvenance
  return Boolean(actual && Object.keys(expected).every(key => actual[key as keyof typeof expected] === expected[key as keyof typeof expected]))
}

export function withoutTranscriptProvenance(state: ClientSessionState): ClientSessionState {
  if (!state.transcriptProvenance) {
    return state
  }
  const { transcriptProvenance: _ignored, ...rest } = state
  return rest
}

export function invalidatePersistedDisplayTranscriptAuthority(state: ClientSessionState): ClientSessionState {
  return { ...state, transcriptAuthorityEpoch: (state.transcriptAuthorityEpoch ?? 0) + 1, transcriptProvenance: undefined }
}

export function suppressTranscriptForView(state: ClientSessionState, suppress: boolean): ClientSessionState {
  return !suppress || state.messages.length === 0 ? state : { ...state, messages: [] }
}

export function shouldPaintPersistedTranscript(prefetchResolved: boolean, resumeIsCurrent: boolean): boolean {
  return prefetchResolved && resumeIsCurrent
}
