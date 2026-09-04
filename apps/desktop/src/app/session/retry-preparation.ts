import type { ClientSessionState } from '@/app/types'
import type { SessionResumeRetryResponse } from '@/types/hermes'

type PreparationRequester = (
  method: string,
  params?: Record<string, unknown>,
  timeoutMs?: number
) => Promise<unknown>

type SessionStateUpdater = (
  runtimeSessionId: string,
  updater: (state: ClientSessionState) => ClientSessionState
) => unknown

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
    preparation: response.preparation
  }))
}
