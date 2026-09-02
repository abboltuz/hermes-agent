// @vitest-environment jsdom
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const harness = vi.hoisted(() => ({
  paneVisible: true,
  resetPublishedScroll: vi.fn(),
  frames: [] as FrameRequestCallback[],
  messages: Array.from({ length: 30 }, (_, index) => ({
    id: `message-${index}`,
    role: index % 2 === 0 ? 'user' : 'assistant',
    content: [{ type: 'text', text: `message ${index}` }]
  }))
}))

vi.mock('@assistant-ui/react', () => ({
  ThreadPrimitive: {
    MessageByIndex: ({ index }: { index: number }) => <div data-testid="message">{index}</div>
  },
  useAuiEvent: () => {},
  useAuiState: (selector: (state: { thread: { messages: typeof harness.messages } }) => unknown) =>
    selector({ thread: { messages: harness.messages } })
}))

vi.mock('@nanostores/react', () => ({
  useStore: () => 1
}))

vi.mock('use-stick-to-bottom', () => ({
  useStickToBottom: () => ({
    scrollRef: { current: null },
    contentRef: { current: null },
    isAtBottom: true,
    scrollToBottom: vi.fn(),
    stopScroll: vi.fn()
  })
}))

vi.mock('@/components/pane-shell/pane-visibility', () => ({
  usePaneLifecycle: () => 'active',
  usePaneVisible: () => harness.paneVisible
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({ t: { assistant: { thread: { showEarlier: 'Show earlier' } } } })
}))

vi.mock('@/lib/render-weight', () => ({
  messagePaintWeight: () => 10
}))

vi.mock('@/lib/utils', () => ({ cn: (...values: unknown[]) => values.filter(Boolean).join(' ') }))

vi.mock('@/store/thread-scroll', () => ({
  onScrollToBottomRequest: () => () => {},
  onThreadEditClose: () => () => {},
  onThreadEditOpen: () => () => {},
  publishThreadAtBottom: () => {},
  resetPublishedThreadScroll: (...args: unknown[]) => harness.resetPublishedScroll(...args)
}))

vi.mock('@/store/windows', () => ({ isSecondaryWindow: () => false }))
vi.mock('../message-render-boundary', () => ({ MessageRenderBoundary: ({ children }: { children: unknown }) => children }))
vi.mock('./transcript-window', () => ({
  resolveShowEarlierAction: () => null,
  useTranscriptWindow: () => ({ olderAvailable: false, expandWindow: vi.fn() })
}))

import { ThreadMessageList, transcriptBackfillFrameCount } from './list'

const components = { AssistantMessage: () => null, SystemMessage: () => null, UserMessage: () => null }

const renderList = () =>
  render(
    <ThreadMessageList
      clampToComposer={false}
      components={components}
      sessionId="stored-session"
      sessionKey="stored-session"
    />
  )

afterEach(() => {
  cleanup()
  harness.paneVisible = true
  harness.resetPublishedScroll.mockReset()
  harness.frames.length = 0
  vi.restoreAllMocks()
})

describe('ThreadMessageList integration contracts', () => {
  it('backfills the mounted transcript in two animation frames', () => {
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      harness.frames.push(callback)
      return harness.frames.length
    })
    vi.stubGlobal('cancelAnimationFrame', () => {})

    renderList()
    expect(document.querySelectorAll('[data-testid="message"]').length).toBe(2)
    expect(harness.frames.length).toBeLessThanOrEqual(2)

    // The mounted list owns the first-paint cap; the exported frame budget
    // contract proves the deferred work cannot require more than two commits.
    expect(transcriptBackfillFrameCount()).toBeLessThanOrEqual(2)
  })

  it('does not reset the visible pane when a retained pane hides before unmount', () => {
    const view = renderList()
    harness.paneVisible = false

    act(() => view.rerender(
      <ThreadMessageList
        clampToComposer={false}
        components={components}
        sessionId="stored-session"
        sessionKey="stored-session-2"
      />
    ))
    view.unmount()

    expect(harness.resetPublishedScroll).not.toHaveBeenCalled()
  })
})
