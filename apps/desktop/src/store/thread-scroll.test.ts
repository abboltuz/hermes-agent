import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $threadJumpButtonVisible,
  onScrollToBottomRequest,
  publishThreadAtBottom,
  requestScrollToBottom,
  resetPublishedThreadScroll
} from './thread-scroll'

afterEach(() => {
  $threadJumpButtonVisible.set(false)
  vi.restoreAllMocks()
})

describe('session-scoped thread scroll bridge', () => {
  it('routes a request only to the target session', () => {
    const first = vi.fn()
    const second = vi.fn()
    const disposeFirst = onScrollToBottomRequest(first, 'session-a')
    const disposeSecond = onScrollToBottomRequest(second, 'session-b')

    requestScrollToBottom('session-a')

    expect(first).toHaveBeenCalledOnce()
    expect(second).not.toHaveBeenCalled()
    disposeFirst()
    disposeSecond()
  })

  it('lets only the visible pane publish and reset composer scroll state', () => {
    publishThreadAtBottom(false, { paneVisible: false })
    expect($threadJumpButtonVisible.get()).toBe(false)

    publishThreadAtBottom(false, { paneVisible: true })
    expect($threadJumpButtonVisible.get()).toBe(true)

    resetPublishedThreadScroll({ paneVisible: false })
    expect($threadJumpButtonVisible.get()).toBe(true)
    resetPublishedThreadScroll({ paneVisible: true })
    expect($threadJumpButtonVisible.get()).toBe(false)
  })
})
