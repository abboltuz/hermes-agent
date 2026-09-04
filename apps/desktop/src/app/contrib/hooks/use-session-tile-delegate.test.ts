import { renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesModule from '@/hermes'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import { setSessionOwnerHint, setSessions } from '@/store/session'
import { sessionTileDelegate } from '@/store/session-states'
import { deferred } from '@/test/deferred'
import type { SessionInfo } from '@/types/hermes'

import { useSessionTileDelegate } from './use-session-tile-delegate'

vi.mock('@/hermes', async importActual => ({
  ...(await importActual<typeof HermesModule>()),
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: '' }))
}))
vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(),
  requestGatewayForProfile: vi.fn()
}))

const { getLatestSessionMessages } = await import('@/hermes')
const { requestGatewayForAgent, requestGatewayForProfile } = await import('@/store/gateway')

const row = (over: Partial<SessionInfo>): SessionInfo =>
  ({
    ended_at: null,
    id: 'live',
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    profile: 'default',
    source: null,
    started_at: 0,
    title: null,
    ...over
  }) as SessionInfo

function renderTile(
  requestGateway: ReturnType<typeof vi.fn>,
  refs?: {
    runtimeIdByStoredSessionIdRef?: { current: Map<string, string> }
    sessionStateByRuntimeIdRef?: { current: Map<string, unknown> }
    updateSessionState?: ReturnType<typeof vi.fn>
  }
) {
  renderHook(() =>
    useSessionTileDelegate({
      archiveSession: vi.fn(async () => undefined),
      branchStoredSession: vi.fn(async () => undefined),
      executeSlashCommand: vi.fn(async () => undefined) as never,
      removeSession: vi.fn(async () => undefined),
      requestGateway: requestGateway as never,
      runtimeIdByStoredSessionIdRef: (refs?.runtimeIdByStoredSessionIdRef ?? { current: new Map() }) as never,
      sessionStateByRuntimeIdRef: (refs?.sessionStateByRuntimeIdRef ?? { current: new Map() }) as never,
      updateSessionState: (refs?.updateSessionState ?? vi.fn()) as never
    })
  )
}

describe('useSessionTileDelegate resumeTile', () => {
  beforeEach(() => {
    setSessions([])
    vi.mocked(getLatestSessionMessages).mockClear()
  })

  afterEach(() => {
    setSessions([])
  })

  it.each([false, true])('keeps history available when runtime fails (runtime first: %s)', async runtimeFirst => {
    setSessions([row({ id: 'readable', profile: 'owner' })])
    const history = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()
    const runtime = deferred<{ session_id: string }>()
    vi.mocked(getLatestSessionMessages).mockReturnValueOnce(history.promise)
    vi.mocked(requestGatewayForProfile).mockReturnValueOnce(runtime.promise)
    const publish = vi.fn()
    renderTile(vi.fn())

    const opening = sessionTileDelegate()!.resumeTile('readable', {
      publish,
      current: () => null,
      isCurrent: () => true
    })

    const settled = opening.catch(error => error)
    await waitFor(() => expect(requestGatewayForProfile).toHaveBeenCalled())
    const failure = new Error('request timed out: session.resume')

    if (runtimeFirst) {
      runtime.reject(failure)
      // Runtime failure is surfaced immediately; a slow archive read cannot
      // hold the retry state hostage.
      expect(await settled).toBe(failure)
    }

    history.resolve({
      session_id: 'readable',
      messages: [{ id: 7, role: 'user', content: 'durable history' }]
    } as never)
    await waitFor(() => expect(publish).toHaveBeenCalledOnce())
    expect(publish.mock.calls[0][0][0].rowId).toBe(7)
    expect(publish.mock.calls[0][1]).toBe('owner')

    if (!runtimeFirst) {
      runtime.reject(failure)
      expect(await settled).toBe(failure)
    }
  })

  it('binds a ready runtime without waiting for a slow archive read', async () => {
    setSessions([row({ id: 'runtime-ready', profile: 'owner' })])
    const history = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()
    vi.mocked(getLatestSessionMessages).mockReturnValueOnce(history.promise)
    vi.mocked(requestGatewayForProfile).mockResolvedValueOnce({
      session_id: 'live-ready',
      messages: [{ row_id: 9, role: 'assistant', content: 'resume tail' }]
    } as never)
    const updateSessionState = vi.fn()
    renderTile(vi.fn(), { updateSessionState })

    await expect(sessionTileDelegate()!.resumeTile('runtime-ready')).resolves.toBe('live-ready')
    expect(updateSessionState).toHaveBeenCalledOnce()

    history.resolve({ session_id: 'runtime-ready', messages: [] })
  })

  it('carries preview backfill into the runtime without overwriting a newer live tail', async () => {
    setSessions([row({ id: 'backfill', profile: 'owner' })])
    const runtime = deferred<{ session_id: string }>()
    vi.mocked(requestGatewayForProfile).mockReturnValueOnce(runtime.promise)
    vi.mocked(getLatestSessionMessages).mockResolvedValueOnce({
      session_id: 'backfill',
      messages: [{ id: 2, role: 'user', content: 'latest' }]
    } as never)
    let preview: ChatMessage[] | null = null
    const updateSessionState = vi.fn()
    renderTile(vi.fn(), { updateSessionState })

    const opening = sessionTileDelegate()!.resumeTile('backfill', {
      publish: messages => {
        preview = messages
      },
      current: () => preview,
      isCurrent: () => true
    })

    await waitFor(() => expect(preview).not.toBeNull())
    const older = toChatMessages([{ id: 1, role: 'user', content: 'older' }] as never)
    preview = [...older, ...preview!]
    runtime.resolve({ session_id: 'live-backfill' })
    await opening
    const update = updateSessionState.mock.calls[0][1]
    expect(update({ messages: [] }).messages.map((message: ChatMessage) => message.rowId)).toEqual([1, 2])

    const live = toChatMessages([
      { id: 2, role: 'user', content: 'latest' },
      { id: 3, role: 'assistant', content: 'new output' }
    ] as never)

    expect(update({ messages: live }).messages.map((message: ChatMessage) => message.rowId)).toEqual([1, 2, 3])
  })

  it('does not bind an obsolete preview attempt into the runtime cache', async () => {
    setSessions([row({ id: 'obsolete', profile: 'owner' })])
    vi.mocked(requestGatewayForProfile).mockResolvedValueOnce({ session_id: 'old-runtime' } as never)
    const updateSessionState = vi.fn()
    renderTile(vi.fn(), { updateSessionState })
    await sessionTileDelegate()!.resumeTile('obsolete', {
      publish: vi.fn(),
      current: () => null,
      isCurrent: () => false
    })
    expect(updateSessionState).not.toHaveBeenCalled()
  })

  it('carries the owning profile into a cold tile resume so it cannot fork profiles', async () => {
    // A tile opens a session owned by another profile. Resuming without the
    // profile lets the gateway fall back to the launch-profile DB and clone the
    // conversation into the wrong profile (#67603). The owning profile must ride
    // both the transcript prefetch and the resume RPC.
    setSessions([row({ id: 'stored-x', profile: 'ai-engineer' })])

    const requestGateway = vi.fn(async (method: string) =>
      method === 'session.resume' ? ({ session_id: 'runtime-1' } as never) : ({} as never)
    )

    vi.mocked(requestGatewayForProfile).mockResolvedValueOnce({ session_id: 'runtime-1' } as never)

    renderTile(requestGateway)
    const runtimeId = await sessionTileDelegate()!.resumeTile('stored-x')

    expect(runtimeId).toBe('runtime-1')
    expect(getLatestSessionMessages).toHaveBeenCalledWith('stored-x', 'ai-engineer')
    expect(requestGatewayForProfile).toHaveBeenCalledWith(
      'ai-engineer',
      'session.resume',
      {
        session_id: 'stored-x',
        cols: 96,
        profile: 'ai-engineer',
        omit_messages: true,
        defer_history: true,
        source: 'desktop'
      },
      undefined,
      undefined
    )
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('resolves and carries a default-profile session explicitly', async () => {
    setSessions([row({ id: 'stored-y', profile: 'default' })])

    const requestGateway = vi.fn(async () => ({}) as never)

    // #92961: a known owner is ALWAYS routed through the profile router —
    // even 'default' — never dispatched on the ambient socket.
    vi.mocked(requestGatewayForProfile).mockResolvedValueOnce({ session_id: 'runtime-2' } as never)

    renderTile(requestGateway)
    const runtimeId = await sessionTileDelegate()!.resumeTile('stored-y')

    expect(runtimeId).toBe('runtime-2')
    expect(requestGatewayForProfile).toHaveBeenCalledWith(
      'default',
      'session.resume',
      {
        session_id: 'stored-y',
        cols: 96,
        profile: 'default',
        omit_messages: true,
        defer_history: true,
        source: 'desktop'
      },
      undefined,
      undefined
    )
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('carries a session row connection owner into a same-named tile resume', async () => {
    setSessions([row({ connection_id: 'source-b', id: 'stored-shared', profile: 'default' })])

    const ambientRequest = vi.fn(async () => ({}) as never)
    vi.mocked(requestGatewayForAgent).mockResolvedValueOnce({ session_id: 'runtime-shared' } as never)

    renderTile(ambientRequest)
    const runtimeId = await sessionTileDelegate()!.resumeTile('stored-shared')

    expect(runtimeId).toBe('runtime-shared')
    expect(requestGatewayForAgent).toHaveBeenCalledWith('source-b', 'default', 'session.resume', {
      session_id: 'stored-shared',
      cols: 96,
      omit_messages: true,
      defer_history: true,
      source: 'desktop',
      profile: 'default'
    })
    expect(ambientRequest).not.toHaveBeenCalled()
  })

  it('routes a Bot tile prefetch and resume through its exact connection owner', async () => {
    const route = {
      connectionId: 'barry',
      mode: 'remote' as const,
      profile: 'oxcoder',
      targetProfile: 'backend-oxcoder'
    }

    setSessionOwnerHint('stored-remote', route)
    vi.mocked(requestGatewayForAgent).mockResolvedValueOnce({ session_id: 'runtime-remote' } as never)
    const ambientRequest = vi.fn(async () => ({}) as never)

    renderTile(ambientRequest)
    const runtimeId = await sessionTileDelegate()!.resumeTile('stored-remote')

    expect(runtimeId).toBe('runtime-remote')
    expect(getLatestSessionMessages).toHaveBeenCalledWith('stored-remote', {
      connectionId: 'barry',
      profile: 'backend-oxcoder'
    })
    expect(requestGatewayForAgent).toHaveBeenCalledWith('barry', 'oxcoder', 'session.resume', {
      session_id: 'stored-remote',
      cols: 96,
      omit_messages: true,
      defer_history: true,
      source: 'desktop',
      profile: 'backend-oxcoder'
    })
    expect(ambientRequest).not.toHaveBeenCalled()
  })

  it('reuses a warm binding that still carries a transcript', async () => {
    const stateA = { busy: false, messages: [{ id: 'm1' }], storedSessionId: 'stored-a' }
    const runtimeIdByStoredSessionIdRef = { current: new Map([['stored-a', 'runtime-a']]) }
    const sessionStateByRuntimeIdRef = { current: new Map([['runtime-a', stateA]]) }
    const requestGateway = vi.fn(async () => ({}) as never)

    renderTile(requestGateway, { runtimeIdByStoredSessionIdRef, sessionStateByRuntimeIdRef })
    const runtimeId = await sessionTileDelegate()!.resumeTile('stored-a')

    expect(runtimeId).toBe('runtime-a')
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('falls through to a real resume when the warm binding has no transcript (post-wake empty tile)', async () => {
    // Sleep/wake regression: a released/stale cached state (messages: []) must
    // NOT satisfy the warm path — reusing it re-bound the tile to a dead
    // runtime id and painted the pane permanently empty.
    setSessions([row({ id: 'stored-b', profile: 'default' })])

    const staleState = { busy: false, messages: [], storedSessionId: 'stored-b' }
    const runtimeIdByStoredSessionIdRef = { current: new Map([['stored-b', 'runtime-dead']]) }
    const sessionStateByRuntimeIdRef = { current: new Map([['runtime-dead', staleState]]) }

    const requestGateway = vi.fn(async () => ({}) as never)

    vi.mocked(requestGatewayForProfile).mockResolvedValueOnce({ session_id: 'runtime-fresh' } as never)

    renderTile(requestGateway, { runtimeIdByStoredSessionIdRef, sessionStateByRuntimeIdRef })
    const runtimeId = await sessionTileDelegate()!.resumeTile('stored-b')

    expect(runtimeId).toBe('runtime-fresh')
    expect(requestGatewayForProfile).toHaveBeenCalledWith(
      'default',
      'session.resume',
      {
        session_id: 'stored-b',
        cols: 96,
        profile: 'default',
        omit_messages: true,
        defer_history: true,
        source: 'desktop'
      },
      undefined,
      undefined
    )
  })

  it('hydrates the tile model and provider from resume info', async () => {
    setSessions([row({ id: 'stored-model', profile: 'default' })])

    const updateSessionState = vi.fn()

    vi.mocked(requestGatewayForProfile).mockResolvedValueOnce({
      info: { fast: true, model: 'gpt-5', provider: 'openai', reasoning_effort: 'high', running: false },
      session_id: 'runtime-model'
    } as never)

    renderTile(vi.fn(), { updateSessionState })
    const runtimeId = await sessionTileDelegate()!.resumeTile('stored-model')

    expect(runtimeId).toBe('runtime-model')
    expect(updateSessionState).toHaveBeenCalled()

    const updater = updateSessionState.mock.calls[0][1] as (state: { messages: unknown[] }) => Record<string, unknown>
    const next = updater({ messages: [] })

    expect(next.model).toBe('gpt-5')
    expect(next.provider).toBe('openai')
    expect(next.reasoningEffort).toBe('high')
    expect(next.fast).toBe(true)
  })

  it('invalidateRuntimeBindings clears the stored→runtime map so tiles re-resume after reconnect', async () => {
    setSessions([row({ id: 'stored-c', profile: 'default' })])

    const liveState = { busy: false, messages: [{ id: 'm1' }], storedSessionId: 'stored-c' }
    const runtimeIdByStoredSessionIdRef = { current: new Map([['stored-c', 'runtime-dead']]) }
    const sessionStateByRuntimeIdRef = { current: new Map([['runtime-dead', liveState]]) }

    const requestGateway = vi.fn(async () => ({}) as never)

    vi.mocked(requestGatewayForProfile).mockResolvedValueOnce({ session_id: 'runtime-fresh' } as never)

    renderTile(requestGateway, { runtimeIdByStoredSessionIdRef, sessionStateByRuntimeIdRef })

    // Gateway reconnect (what resetTileRuntimeBindings calls on wake):
    sessionTileDelegate()!.invalidateRuntimeBindings!()
    expect(runtimeIdByStoredSessionIdRef.current.size).toBe(0)

    // The next resume goes cold instead of reusing the dead binding.
    const runtimeId = await sessionTileDelegate()!.resumeTile('stored-c')
    expect(runtimeId).toBe('runtime-fresh')
  })
})

describe('useSessionTileDelegate retireBusyClaim', () => {
  it('retires a stale busy claim through the session-state write path (#93059)', () => {
    const busyState = { awaitingResponse: true, busy: true, messages: [{ id: 'm1' }], storedSessionId: 'stored-d' }
    const sessionStateByRuntimeIdRef = { current: new Map([['runtime-dead', busyState]]) }
    const updateSessionState = vi.fn()

    renderTile(
      vi.fn(async () => ({}) as never),
      { sessionStateByRuntimeIdRef, updateSessionState }
    )

    expect(sessionTileDelegate()!.retireBusyClaim!('runtime-dead')).toBe(true)
    expect(updateSessionState).toHaveBeenCalledWith('runtime-dead', expect.any(Function))

    // The updater is the downgrade: busy/awaiting off, everything else intact.
    const updater = updateSessionState.mock.calls[0][1] as (state: typeof busyState) => typeof busyState

    expect(updater(busyState)).toEqual({ ...busyState, awaitingResponse: false, busy: false })
  })

  it('reports a miss instead of minting a cache entry for a runtime it never held', () => {
    // No phantoms: updateSessionState mints a state for any id it is handed,
    // and prune never collects a transcript-less entry — so a miss must not
    // reach the write path; the store retires its own mirror instead.
    const idle = { awaitingResponse: false, busy: false, messages: [{ id: 'm1' }], storedSessionId: 'stored-e' }
    const sessionStateByRuntimeIdRef = { current: new Map([['runtime-idle', idle]]) }
    const updateSessionState = vi.fn()

    renderTile(
      vi.fn(async () => ({}) as never),
      { sessionStateByRuntimeIdRef, updateSessionState }
    )

    expect(sessionTileDelegate()!.retireBusyClaim!('runtime-unknown')).toBe(false)
    expect(sessionTileDelegate()!.retireBusyClaim!('runtime-idle')).toBe(false)
    expect(updateSessionState).not.toHaveBeenCalled()
  })
})

describe('useSessionTileDelegate interruptSession', () => {
  beforeEach(() => {
    setSessions([])
  })

  afterEach(async () => {
    setSessions([])
    const { clearSessionRecentlyInterrupted } = await import('../../session/hooks/use-prompt-actions/utils')
    clearSessionRecentlyInterrupted()
  })

  it('marks the session recently interrupted so a quick tile edit/resend still interrupt-firsts (#83855)', async () => {
    const { isSessionRecentlyInterrupted } = await import('../../session/hooks/use-prompt-actions/utils')

    const requestGateway = vi.fn(async () => ({}) as never)

    renderTile(requestGateway)
    await sessionTileDelegate()!.interruptSession('runtime-tile-1')

    expect(requestGateway).toHaveBeenCalledWith('session.interrupt', { session_id: 'runtime-tile-1' })
    // Same 3s cooldown the primary chat's Stop sets: busy reads false while the
    // gateway winds down, so the rewind path must still interrupt-first.
    expect(isSessionRecentlyInterrupted('runtime-tile-1')).toBe(true)
  })
})
