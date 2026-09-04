import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { stubThreadEnvironment, stubThreadViewportSize } from '@/components/assistant-ui/test-utils'
import { toChatMessages } from '@/lib/chat-messages'
import { $gatewayState, setSessions } from '@/store/session'
import { $sessionTiles, type SessionTileDelegate, setSessionTileDelegate } from '@/store/session-states'
import { $transcriptTailBySessionId, recordTranscriptTail } from '@/store/transcript-tail'
import { deferred } from '@/test/deferred'
import { makeSessionInfo } from '@/test/session-info'

import { SessionTilePane } from './session-tile'

vi.mock('@/hermes', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  getOlderSessionMessages: vi.fn()
}))

// The pane owns history/resume orchestration; the shared transcript renderer
// has its own component suite. Keep this test on the pane boundary so an
// upstream assistant-ui test-runtime snapshot bug cannot turn a successful
// orchestration assertion into an unrelated unhandled cleanup error.
vi.mock('.', async () => {
  const { useStore } = await import('@nanostores/react')
  const { chatMessageText } = await import('@/lib/chat-messages')
  const { $transcriptTailBySessionId } = await import('@/store/transcript-tail')
  const { useSessionView } = await import('./session-view')
  const { backfillOlderTranscriptPage, transcriptBackfillAvailable } = await import('./transcript-backfill')

  return {
    ChatView: () => null,
    StoredSessionTranscript: ({
      historyProfile,
      onOlderPage
    }: {
      historyProfile?: typeof scope
      onOlderPage: (messages: ReturnType<typeof useStore>) => void
    }) => {
      const view = useSessionView()
      const messages = useStore(view.$messages)
      const storedId = useStore(view.$storedId)
      useStore($transcriptTailBySessionId)
      const olderAvailable = transcriptBackfillAvailable(storedId, historyProfile)

      return (
        <div>
          {olderAvailable && storedId && (
            <button
              onClick={() =>
                void backfillOlderTranscriptPage({
                  applyOlderPage: onOlderPage as never,
                  isCurrent: () => view.$storedId.get() === storedId && view.$runtimeId.get() === null,
                  profile: historyProfile,
                  storedSessionId: storedId
                })
              }
            >
              Show earlier
            </button>
          )}
          {messages.map(message => (
            <div key={message.id}>{chatMessageText(message)}</div>
          ))}
        </div>
      )
    }
  }
})

const { getOlderSessionMessages } = await import('@/hermes')
stubThreadEnvironment()
stubThreadViewportSize()

const scope = { connectionId: 'remote-owner', profile: 'owner' }
const human = { origin_kind: 'human_user', turn_kind: 'prompt', trust_kind: 'user_authorized' } as const

const page = {
  session_id: 'chat',
  messages: [{ id: 2, role: 'assistant', content: 'A preserved answer' }],
  pagination: { limit: 1, offset: 0, order: 'latest', returned: 1 }
} as const

function installResume(resumeTile: SessionTileDelegate['resumeTile']) {
  setSessionTileDelegate({
    archiveSession: vi.fn(),
    branchSession: vi.fn(),
    deleteSession: vi.fn(),
    executeSlash: vi.fn(),
    interruptSession: vi.fn(),
    submitToSession: vi.fn(),
    updateSession: vi.fn(),
    resumeTile
  })
}

describe('tile history without an agent runtime', () => {
  beforeEach(() => {
    $gatewayState.set('open')
    $sessionTiles.set([{ storedSessionId: 'chat', ownerRoute: scope }])
    setSessions([makeSessionInfo({ id: 'chat', profile: 'owner' })])
    $transcriptTailBySessionId.set({})
  })
  afterEach(() => {
    $sessionTiles.set([])
    $transcriptTailBySessionId.set({})
    setSessions([])
  })

  it('reads and backfills history after runtime failure, and keeps it visible during retry', async () => {
    const runtime = deferred<string>()
    const retry = deferred<string>()
    let attempt: Parameters<SessionTileDelegate['resumeTile']>[1]
    let attempts = 0

    const resume = vi.fn<SessionTileDelegate['resumeTile']>((_id, transcript) => {
      attempt = transcript

      return ++attempts === 1 ? runtime.promise : retry.promise
    })

    installResume(resume)
    render(
      <MemoryRouter>
        <SessionTilePane storedSessionId="chat" />
      </MemoryRouter>
    )
    await waitFor(() => expect(resume).toHaveBeenCalledOnce())
    act(() => {
      recordTranscriptTail('chat', page as never, scope)
      attempt!.publish(toChatMessages(page.messages as never), scope)
    })
    expect(await screen.findByText('A preserved answer')).toBeTruthy()
    expect(screen.queryByRole('button', { name: /^edit$/i })).toBeNull()
    expect(screen.queryByRole('textbox')).toBeNull()
    await act(async () => runtime.reject(new Error('agent resume timed out')))
    expect(await screen.findByText(/agent resume timed out/)).toBeTruthy()
    expect(screen.getByText('A preserved answer')).toBeTruthy()

    vi.mocked(getOlderSessionMessages).mockResolvedValueOnce({
      session_id: 'chat',
      messages: [{ id: 1, role: 'user', content: 'An older exact question', ...human }],
      pagination: { limit: 120, offset: 1, order: 'latest', returned: 1 }
    } as never)
    fireEvent.click(screen.getByRole('button', { name: /show earlier/i }))
    await waitFor(() => expect(getOlderSessionMessages).toHaveBeenCalled())
    expect(await screen.findByText('An older exact question')).toBeTruthy()
    expect(getOlderSessionMessages).toHaveBeenCalledWith('chat', scope, 1)

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(resume).toHaveBeenCalledTimes(2))
    expect(screen.getByText('A preserved answer')).toBeTruthy()
    expect(screen.getByText('An older exact question')).toBeTruthy()
    expect(attempt!.current()!.map(message => message.rowId)).toEqual([1, 2])
  })

  it('promotes a late history result after runtime failure into the readable pane', async () => {
    const runtime = deferred<string>()
    let attempt: Parameters<SessionTileDelegate['resumeTile']>[1]

    installResume((_id, transcript) => {
      attempt = transcript

      return runtime.promise
    })
    render(
      <MemoryRouter>
        <SessionTilePane storedSessionId="chat" />
      </MemoryRouter>
    )
    await waitFor(() => expect(attempt).toBeDefined())
    await act(async () => runtime.reject(new Error('agent resume timed out')))
    expect(await screen.findByText(/agent resume timed out/)).toBeTruthy()
    expect(attempt!.isCurrent()).toBe(true)

    act(() => attempt!.publish(toChatMessages(page.messages as never), scope))
    expect(await screen.findByText('A preserved answer')).toBeTruthy()
    expect(screen.getByText(/agent resume timed out/)).toBeTruthy()
  })

  it('ignores a late history result after the tile has unmounted', async () => {
    let attempt: Parameters<SessionTileDelegate['resumeTile']>[1]
    installResume((_id, transcript) => {
      attempt = transcript

      return new Promise(() => {})
    })

    const mounted = render(
      <MemoryRouter>
        <SessionTilePane storedSessionId="chat" />
      </MemoryRouter>
    )

    await waitFor(() => expect(attempt).toBeDefined())
    mounted.unmount()
    act(() => attempt!.publish(toChatMessages(page.messages as never), scope))
    expect(attempt!.isCurrent()).toBe(false)
    expect(attempt!.current()).toBeNull()
  })
})
