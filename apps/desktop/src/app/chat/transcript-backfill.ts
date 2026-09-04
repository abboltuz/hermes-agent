/**
 * ON-DEMAND OLDER-PAGE BACKFILL for the transcript window.
 *
 * Tail hydration (`getLatestSessionMessages`) loads only the newest page of a
 * session. "Show earlier" first pages the DOM budget, then the in-memory store
 * window — and when the whole in-memory transcript is materialized but the
 * REST hydration was truncated (`transcript-tail` bookkeeping), this module
 * fetches the next older page and prepends it to the session store.
 *
 * Offsets follow the backend's `order: 'latest'` semantics: measured back
 * from the NEWEST persisted row. Rows persisted after hydration shift that
 * origin, so a fetched page can overlap rows we already hold — the prepend
 * dedupes by durable row id (falling back to the rendered message id) and
 * the offset still advances by the fetched count, which self-corrects the
 * drift on the next page.
 */

import { getOlderSessionMessages } from '@/hermes'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import { recordTranscriptBackfillPage, type TranscriptProfileScope, transcriptTailState } from '@/store/transcript-tail'

/** Older rows likely exist beyond what the in-memory store holds. */
export function transcriptBackfillAvailable(
  storedSessionId: null | string | undefined,
  profile?: TranscriptProfileScope
): boolean {
  return Boolean(transcriptTailState(storedSessionId, profile)?.possiblyTruncated)
}

/**
 * Prepend an older page onto the in-memory transcript, deduplicating rows the
 * store already holds (offset drift makes overlap normal — see module doc).
 * Preserves reference identity when nothing changes: handing React a fresh
 * array of the same messages re-renders the runtime for nothing.
 */
export function mergeOlderTranscriptPage(existing: ChatMessage[], olderPage: ChatMessage[]): ChatMessage[] {
  // Backfill only makes sense under an already-hydrated tail. An empty store
  // here means the session was swapped or wiped mid-fetch; prepending would
  // paint the older page as the whole conversation.
  if (existing.length === 0 || olderPage.length === 0) {
    return existing
  }

  const fresh = olderPage.filter(message => !existing.some(current => sameTranscriptIdentity(current, message)))

  if (fresh.length === 0) {
    return existing
  }

  return [...fresh, ...existing]
}

function sameTranscriptIdentity(left: ChatMessage, right: ChatMessage): boolean {
  if (left.rowId !== undefined && right.rowId !== undefined) {
    return left.rowId === right.rowId
  }

  if (left.semanticId || right.semanticId) {
    return Boolean(left.semanticId) && left.semanticId === right.semanticId
  }

  return left.id === right.id
}

/**
 * Attach a persisted tail that arrived after a runtime was already bound.
 * Runtime messages win on overlap because they can contain richer streaming
 * state; persisted rows fill the chronological prefix. Exact stored/runtime
 * fencing belongs to the caller — this function only reconciles one session.
 */
export function mergePersistedTailIntoRuntime(
  currentRuntime: ChatMessage[],
  persistedTail: ChatMessage[]
): ChatMessage[] {
  if (persistedTail.length === 0) {
    return currentRuntime
  }

  if (currentRuntime.length === 0) {
    return persistedTail
  }

  // Build a shortest common supersequence. Persisted and runtime are both
  // chronological, but either may omit rows the other carries; prepending all
  // missing persisted rows would reorder a gap inside an overlap. The LCS
  // anchors those gaps, and the runtime object wins at each shared identity so
  // richer live/pending state survives hydration.
  const persistedCount = persistedTail.length
  const runtimeCount = currentRuntime.length
  const widths = runtimeCount + 1
  const lcs = new Uint32Array((persistedCount + 1) * widths)

  for (let persistedIndex = persistedCount - 1; persistedIndex >= 0; persistedIndex -= 1) {
    for (let runtimeIndex = runtimeCount - 1; runtimeIndex >= 0; runtimeIndex -= 1) {
      const offset = persistedIndex * widths + runtimeIndex

      lcs[offset] = sameTranscriptIdentity(persistedTail[persistedIndex], currentRuntime[runtimeIndex])
        ? lcs[(persistedIndex + 1) * widths + runtimeIndex + 1] + 1
        : Math.max(lcs[(persistedIndex + 1) * widths + runtimeIndex], lcs[offset + 1])
    }
  }

  const merged: ChatMessage[] = []
  let persistedIndex = 0
  let runtimeIndex = 0

  while (persistedIndex < persistedCount && runtimeIndex < runtimeCount) {
    if (sameTranscriptIdentity(persistedTail[persistedIndex], currentRuntime[runtimeIndex])) {
      merged.push(currentRuntime[runtimeIndex])
      persistedIndex += 1
      runtimeIndex += 1
    } else if (lcs[(persistedIndex + 1) * widths + runtimeIndex] >= lcs[persistedIndex * widths + runtimeIndex + 1]) {
      merged.push(persistedTail[persistedIndex])
      persistedIndex += 1
    } else {
      merged.push(currentRuntime[runtimeIndex])
      runtimeIndex += 1
    }
  }

  merged.push(...persistedTail.slice(persistedIndex), ...currentRuntime.slice(runtimeIndex))

  return merged
}

/**
 * Re-anchor a refreshed TAIL onto a transcript that has backfilled older
 * pages. Background refreshes and post-turn rehydrates re-read only the
 * newest page; replacing the store with that page outright would silently
 * drop everything "Show earlier" already loaded. Find where the refreshed
 * tail begins inside the previous transcript and keep the older prefix.
 * When no anchor is found (compaction rewrite, different session), the
 * refreshed tail is authoritative — same behavior as before backfill existed.
 */
export function graftRefreshedTailOntoBackfill(refreshedTail: ChatMessage[], previous: ChatMessage[]): ChatMessage[] {
  if (refreshedTail.length === 0 || previous.length === 0) {
    return refreshedTail
  }

  const first = refreshedTail[0]

  const anchor = previous.findIndex(message => sameTranscriptIdentity(message, first))

  if (anchor <= 0) {
    return refreshedTail
  }

  return [...previous.slice(0, anchor), ...refreshedTail]
}

export interface BackfillRequest {
  /** Durable stored session id — the tail bookkeeping key. */
  storedSessionId: string
  /** Owner scope captured when the tail was hydrated. */
  profile?: TranscriptProfileScope
  /** Stale-response guard: called after the fetch resolves; when it reports
   *  false (the user switched sessions mid-flight) the page is discarded and
   *  the bookkeeping is left untouched, mirroring the isCurrentResume()
   *  pattern in use-session-actions. */
  isCurrent: () => boolean
  /** Apply the converted older page to the session's message store. The
   *  callback owns WHERE the messages live (session-state cache vs the global
   *  draft atom) and must merge via `mergeOlderTranscriptPage`. */
  applyOlderPage: (olderPage: ChatMessage[]) => void
}

// One fetch per stored session at a time. Keyed by stored id (not runtime id)
// so a mid-fetch runtime rebind cannot double-fetch the same page.
const inflightByStoredSessionId = new Map<string, Promise<boolean>>()

/** Test-only: drop in-flight guards between cases. */
export function _resetTranscriptBackfillForTests(): void {
  inflightByStoredSessionId.clear()
}

/**
 * Fetch the next older page for a session and prepend it via
 * `applyOlderPage`. Resolves true when a page was applied. Concurrent calls
 * for the same session share one fetch.
 */
export function backfillOlderTranscriptPage(request: BackfillRequest): Promise<boolean> {
  const { profile, storedSessionId } = request
  const inflightKey = JSON.stringify([profile || null, storedSessionId])
  const inflight = inflightByStoredSessionId.get(inflightKey)

  if (inflight) {
    return inflight
  }

  const run = (async () => {
    const tail = transcriptTailState(storedSessionId, profile)

    if (!tail?.possiblyTruncated) {
      return false
    }

    let page

    try {
      page = await getOlderSessionMessages(storedSessionId, tail.profile, tail.nextOffset)
    } catch {
      // Non-fatal: the action stays available and the next click retries.
      return false
    }

    // Session switched while the page was in flight: discard it entirely.
    // The bookkeeping stays untouched so a later re-visit (which re-records
    // the tail on hydration anyway) starts from consistent state.
    if (!request.isCurrent()) {
      return false
    }

    // A response without pagination metadata is a legacy backend that ignored
    // the paging query and returned the FULL transcript one-shot. The merge
    // below prepends whatever prefix the store is missing, and the recorded
    // state marks the session fully loaded so the REST action retires.
    recordTranscriptBackfillPage(storedSessionId, page, profile)
    request.applyOlderPage(toChatMessages(page.messages))

    return true
  })().finally(() => {
    inflightByStoredSessionId.delete(inflightKey)
  })

  inflightByStoredSessionId.set(inflightKey, run)

  return run
}
