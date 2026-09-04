import type { ClientSessionState } from '@/app/types'
import type { SessionPreparation, SessionResumeRetryResponse } from '@/types/hermes'

type PreparationRequester = (
  method: string,
  params?: Record<string, unknown>,
  timeoutMs?: number
) => Promise<unknown>

type SessionStateUpdater = (
  runtimeSessionId: string,
  updater: (state: ClientSessionState) => ClientSessionState
) => unknown

const preparationRank = (preparation: SessionPreparation): number =>
  preparation.status === 'preparing' ? 0 : 1

/** Merge response/event state monotonically. A fast worker can publish a
 * terminal progress event before the original RPC response is applied; that
 * stale `preparing` acknowledgement must not reopen the preparation gate. */
export function mergeSessionPreparation(
  current: SessionPreparation | undefined,
  incoming: SessionPreparation | undefined
): SessionPreparation | undefined {
  if (!incoming) {
    return current
  }

  if (!current) {
    return incoming
  }

  if (incoming.attempt !== current.attempt) {
    return incoming.attempt > current.attempt ? incoming : current
  }

  const currentRank = preparationRank(current)
  const incomingRank = preparationRank(incoming)

  if (incomingRank < currentRank) {
    return current
  }

  if (incomingRank === currentRank && incoming.status !== current.status) {
    return current
  }

  return incoming
}

/** Retry the model-facing history build on the retained runtime. Display
 * history is deliberately untouched: it belongs to the independent archive
 * projection and remains readable throughout this request. */
export async function retryRetainedSessionPreparation(
  requestGateway: PreparationRequester,
  updateSessionState: SessionStateUpdater,
  runtimeSessionId: string
): Promise<void> {
  const response = (await requestGateway('session.resume.retry', {
    session_id: runtimeSessionId
  })) as SessionResumeRetryResponse

  updateSessionState(runtimeSessionId, state => ({
    ...state,
    preparation: mergeSessionPreparation(state.preparation, response.preparation)
  }))
}
