import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ConfirmHost } from '@/components/confirm-host'
import type { AntigravityAccount, AntigravityRpc } from '@/lib/antigravity-rpc'
import type * as AntigravityRpcModule from '@/lib/antigravity-rpc'
import { $confirmRequest } from '@/store/confirm'
import { $activeGatewayProfile } from '@/store/profile'

const { createAntigravityRpc, openExternalLink, requestGateway } = vi.hoisted(() => ({
  createAntigravityRpc: vi.fn(),
  openExternalLink: vi.fn(),
  requestGateway: vi.fn()
}))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

vi.mock('@/lib/antigravity-rpc', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  createAntigravityRpc
}))

vi.mock('@/lib/external-link', () => ({ openExternalLink }))

const account = (suffix: string, patch: Partial<AntigravityAccount> = {}): AntigravityAccount => ({
  enabled: true,
  id: `acct_11111111-1111-4111-8111-11111111111${suffix}`,
  priority: 2,
  ...patch
})

const snapshot = (accounts: AntigravityAccount[]) => ({ accounts })

const oauthStart = {
  authUrl: 'https://accounts.google.com/o/oauth2/v2/auth?client_id=desktop',
  expiresAt: 999_999,
  flow: 'browser_poll',
  pollIntervalMs: 1_000,
  sessionId: '11111111-1111-4111-8111-111111111111',
  status: 'pending'
}

const rpc: Record<keyof AntigravityRpc, ReturnType<typeof vi.fn>> = {
  cancelOAuth: vi.fn(),
  listAccounts: vi.fn(),
  pollOAuth: vi.fn(),
  removeAccount: vi.fn(),
  setAccountEnabled: vi.fn(),
  setAccountPriority: vi.fn(),
  startOAuth: vi.fn()
}

function deferred<T>() {
  let reject!: (reason?: unknown) => void
  let resolve!: (value: T) => void

  const promise = new Promise<T>((innerResolve, innerReject) => {
    resolve = innerResolve
    reject = innerReject
  })

  return { promise, reject, resolve }
}

async function renderAccounts(onConfigSaved?: () => void) {
  const { AntigravityAccounts } = await import('./antigravity-accounts')

  return render(
    <>
      <AntigravityAccounts onConfigSaved={onConfigSaved} />
      <ConfirmHost />
    </>
  )
}

beforeEach(() => {
  $activeGatewayProfile.set('default')
  $confirmRequest.set(null)
  requestGateway.mockReset()
  openExternalLink.mockReset()
  createAntigravityRpc.mockReturnValue(rpc)
  Object.values(rpc).forEach(method => method.mockReset())
  rpc.listAccounts.mockResolvedValue(snapshot([account('1')]))
  rpc.setAccountEnabled.mockResolvedValue(snapshot([account('1', { enabled: false })]))
  rpc.setAccountPriority.mockResolvedValue(snapshot([account('1', { priority: 4 })]))
  rpc.removeAccount.mockResolvedValue(snapshot([]))
  rpc.startOAuth.mockResolvedValue(oauthStart)
  rpc.pollOAuth.mockResolvedValue({ status: 'pending' })
  rpc.cancelOAuth.mockResolvedValue({ status: 'cancelled' })
})

afterEach(() => {
  cleanup()
  $confirmRequest.set(null)
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('AntigravityAccounts', () => {
  it('keeps the verification reason visible until a refreshed successful status replaces it', async () => {
    rpc.listAccounts.mockResolvedValueOnce(
      snapshot([account('1', { enabled: false, status: 'verification_required' })])
    )
    rpc.setAccountEnabled.mockResolvedValueOnce(
      snapshot([account('1', { enabled: true, status: 'verification_required' })])
    )
    await renderAccounts()
    expect(await screen.findByText('Google account verification required')).toBeTruthy()
    const toggle = screen.getByRole('switch', { name: /enable antigravity account/i })
    expect(toggle.getAttribute('aria-checked')).toBe('false')
    fireEvent.click(toggle)
    await waitFor(() => expect(toggle.getAttribute('aria-checked')).toBe('true'))
    expect(screen.getByText('Google account verification required')).toBeTruthy()
    rpc.listAccounts.mockResolvedValueOnce(snapshot([account('1', { enabled: true, status: 'available' })]))
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }))
    expect(await screen.findByText('Enabled')).toBeTruthy()
    expect(screen.queryByText('Google account verification required')).toBeNull()
  })

  it('shows email while mutations continue to use the opaque account id', async () => {
    rpc.listAccounts.mockResolvedValue(snapshot([account('1', { email: 'artem@example.test' })]))
    await renderAccounts()
    expect(await screen.findByText('artem@example.test')).toBeTruthy()
    expect(screen.queryByText(account('1').id)).toBeNull()
    fireEvent.click(screen.getByRole('switch', { name: /artem@example.test/ }))
    await waitFor(() => expect(rpc.setAccountEnabled).toHaveBeenCalledWith(account('1').id, false, expect.anything()))
  })

  it('lists privacy-safe account rows and refreshes under the active profile', async () => {
    await renderAccounts()

    await screen.findByText('Antigravity accounts')
    expect(rpc.listAccounts).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('switch', { name: /enable antigravity account/i })).toBeTruthy()
    expect(screen.getByDisplayValue('2')).toBeTruthy()
    expect(screen.queryByText(/token|fingerprint|email/i)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }))

    await waitFor(() => expect(rpc.listAccounts).toHaveBeenCalledTimes(2))
    expect(createAntigravityRpc).toHaveBeenCalledWith(requestGateway, expect.any(Function))
    expect(createAntigravityRpc.mock.calls[0][1]()).toBe('default')
  })

  it('renders an initial load failure and lets the user retry it', async () => {
    const initialList = deferred<{ accounts: AntigravityAccount[] }>()
    rpc.listAccounts.mockReturnValueOnce(initialList.promise).mockResolvedValueOnce(snapshot([account('1')]))
    await renderAccounts()

    await act(async () => {
      initialList.reject(new Error('private bridge diagnostic'))
    })
    expect(await screen.findByText('Could not load Antigravity accounts.')).toBeTruthy()
    expect(screen.queryByText('Loading…')).toBeNull()

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    })

    expect(await screen.findByRole('switch', { name: /enable antigravity account/i })).toBeTruthy()
    expect(rpc.listAccounts).toHaveBeenCalledTimes(2)
  })

  it('prevents duplicate mutations and preserves the row after a failed enabled update', async () => {
    const write = deferred<{ accounts: AntigravityAccount[] }>()
    rpc.setAccountEnabled.mockReturnValueOnce(write.promise)
    await renderAccounts()
    const toggle = await screen.findByRole('switch', { name: /enable antigravity account/i })

    fireEvent.click(toggle)
    fireEvent.click(toggle)

    expect(rpc.setAccountEnabled).toHaveBeenCalledTimes(1)
    write.resolve(snapshot([account('1', { enabled: false })]))
    await waitFor(() => expect(toggle.getAttribute('aria-checked')).toBe('false'))

    rpc.setAccountEnabled.mockRejectedValueOnce(new Error('private bridge diagnostic'))
    fireEvent.click(toggle)

    await waitFor(() => expect(rpc.setAccountEnabled).toHaveBeenCalledTimes(2))
    expect(screen.getByRole('switch', { name: /enable antigravity account/i }).getAttribute('aria-checked')).toBe(
      'false'
    )
  })

  it('bounds priority input and confirms removal before applying the server snapshot', async () => {
    await renderAccounts()
    const priority = await screen.findByLabelText(/priority for antigravity account/i)

    fireEvent.change(priority, { target: { value: '0' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save priority' }))
    expect(rpc.setAccountPriority).not.toHaveBeenCalled()
    expect(await screen.findByText('Priority must be between 1 and 999999.')).toBeTruthy()

    fireEvent.change(priority, { target: { value: '4' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save priority' }))
    await waitFor(() => expect(rpc.setAccountPriority).toHaveBeenCalledWith(account('1').id, 4, expect.anything()))

    fireEvent.click(screen.getByRole('button', { name: /remove antigravity account/i }))
    expect(rpc.removeAccount).not.toHaveBeenCalled()
    fireEvent.click(await screen.findByRole('button', { name: 'Remove' }))
    await waitFor(() => expect(rpc.removeAccount).toHaveBeenCalledWith(account('1').id, expect.anything()))
    expect(screen.queryByRole('switch', { name: /enable antigravity account/i })).toBeNull()
  })

  it('invalidates model options after successful removal but not a failed removal', async () => {
    const onConfigSaved = vi.fn()
    await renderAccounts(onConfigSaved)
    await screen.findByRole('switch', { name: /enable antigravity account/i })

    fireEvent.click(screen.getByRole('button', { name: /remove antigravity account/i }))
    fireEvent.click(await screen.findByRole('button', { name: 'Remove' }))
    await waitFor(() => expect(onConfigSaved).toHaveBeenCalledTimes(1))

    rpc.listAccounts.mockResolvedValueOnce(snapshot([account('1')]))
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }))
    await screen.findByRole('switch', { name: /enable antigravity account/i })
    rpc.removeAccount.mockRejectedValueOnce(new Error('private bridge diagnostic'))
    fireEvent.click(screen.getByRole('button', { name: /remove antigravity account/i }))
    fireEvent.click(await screen.findByRole('button', { name: 'Remove' }))
    await waitFor(() => expect(rpc.removeAccount).toHaveBeenCalledTimes(2))
    expect(onConfigSaved).toHaveBeenCalledTimes(1)
  })

  it('invalidates model options after approved OAuth only when its active-profile snapshot applies', async () => {
    const onConfigSaved = vi.fn()
    await renderAccounts(onConfigSaved)
    await screen.findByText('Antigravity accounts')
    vi.useFakeTimers()

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Connect account' }))
    })
    rpc.pollOAuth.mockResolvedValueOnce({ status: 'approved' })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000)
      await Promise.resolve()
    })
    expect(onConfigSaved).toHaveBeenCalledTimes(1)
  })

  it('review probe invalidates model options after approval when the account refresh fails', async () => {
    const onConfigSaved = vi.fn()
    await renderAccounts(onConfigSaved)
    await screen.findByText('Antigravity accounts')
    vi.useFakeTimers()

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Connect account' }))
    })
    rpc.listAccounts.mockRejectedValueOnce(new Error('private bridge diagnostic'))
    rpc.pollOAuth.mockResolvedValueOnce({ status: 'approved' })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000)
      await Promise.resolve()
    })

    expect(onConfigSaved).toHaveBeenCalledTimes(1)
    expect(screen.getByText('Could not load Antigravity accounts.')).toBeTruthy()
    expect(screen.queryByText('private bridge diagnostic')).toBeNull()
  })

  it('does not invalidate model options for an approved OAuth result after a profile transition', async () => {
    const onConfigSaved = vi.fn()
    const approval = deferred<{ status: string }>()
    rpc.pollOAuth.mockReturnValueOnce(approval.promise)
    await renderAccounts(onConfigSaved)
    await screen.findByText('Antigravity accounts')
    vi.useFakeTimers()

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Connect account' }))
      await vi.advanceTimersByTimeAsync(1_000)
    })
    expect(rpc.pollOAuth).toHaveBeenCalledTimes(1)

    await act(async () => {
      $activeGatewayProfile.set('other')
      await Promise.resolve()
    })
    expect(rpc.listAccounts).toHaveBeenCalledTimes(2)
    expect($activeGatewayProfile.get()).toBe('other')

    await act(async () => {
      approval.resolve({ status: 'approved' })
    })

    expect(onConfigSaved).not.toHaveBeenCalled()
  })

  it('starts OAuth only through the typed client without rendering a Project ID field and cancels its poll lifecycle', async () => {
    await renderAccounts()
    await screen.findByText('Antigravity accounts')

    expect(screen.queryByLabelText('Project ID')).toBeNull()
    vi.useFakeTimers()
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Connect account' }))
    })

    expect(rpc.startOAuth).toHaveBeenCalledWith(expect.anything())
    expect(openExternalLink).toHaveBeenCalledWith(oauthStart.authUrl)
    expect(requestGateway).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeTruthy()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000)
    })
    expect(rpc.pollOAuth).toHaveBeenCalledWith(oauthStart.sessionId, expect.anything())

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    })
    expect(rpc.cancelOAuth).toHaveBeenCalledWith(oauthStart.sessionId)
  })

  it('starts OAuth without a project_id own-property', async () => {
    const realRpc = await vi.importActual<typeof AntigravityRpcModule>('@/lib/antigravity-rpc')
    createAntigravityRpc.mockImplementation((request, getActiveProfile) =>
      realRpc.createAntigravityRpc(request, getActiveProfile)
    )
    requestGateway.mockImplementation(async method => {
      if (method === 'antigravity.accounts.list') {
        return { accounts: snapshot([account('1')]) }
      }

      if (method === 'antigravity.oauth.start') {
        return {
          auth_url: oauthStart.authUrl,
          expires_at: oauthStart.expiresAt,
          flow: oauthStart.flow,
          poll_interval_ms: oauthStart.pollIntervalMs,
          session_id: oauthStart.sessionId,
          status: oauthStart.status
        }
      }

      throw new Error('unexpected gateway method')
    })
    await renderAccounts()
    await screen.findByText('Antigravity accounts')

    fireEvent.click(screen.getByRole('button', { name: 'Connect account' }))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'antigravity.oauth.start',
        { profile: 'default' },
        undefined,
        expect.anything()
      )
    )
    const oauthStartCall = requestGateway.mock.calls.find(([method]) => method === 'antigravity.oauth.start')
    expect(oauthStartCall).toBeDefined()

    const [, params] = oauthStartCall!
    expect(params).toStrictEqual({ profile: 'default' })
    expect(Object.hasOwn(params, 'project_id')).toBe(false)
    expect(openExternalLink).toHaveBeenCalledWith(oauthStart.authUrl)
  })

  it('prevents a duplicate OAuth start while the first start is pending', async () => {
    const start = deferred<typeof oauthStart>()
    rpc.startOAuth.mockReturnValueOnce(start.promise)
    await renderAccounts()
    await screen.findByText('Antigravity accounts')

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Connect account' }))
      fireEvent.click(screen.getByRole('button', { name: 'Connect account' }))
    })

    expect(rpc.startOAuth).toHaveBeenCalledTimes(1)
  })

  it('cancels OAuth when the settings surface unmounts', async () => {
    const view = await renderAccounts()
    await screen.findByText('Antigravity accounts')

    fireEvent.click(screen.getByRole('button', { name: 'Connect account' }))
    await waitFor(() => expect(rpc.startOAuth).toHaveBeenCalledTimes(1))
    view.unmount()

    expect(rpc.cancelOAuth).toHaveBeenCalledWith(oauthStart.sessionId)
  })

  it('cancels a pending OAuth start without opening or polling after unmount', async () => {
    const start = deferred<typeof oauthStart>()
    rpc.startOAuth.mockReturnValueOnce(start.promise)
    const view = await renderAccounts()
    await screen.findByText('Antigravity accounts')

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Connect account' }))
    })
    await waitFor(() => expect(rpc.startOAuth).toHaveBeenCalledTimes(1))
    view.unmount()

    await act(async () => {
      start.resolve(oauthStart)
    })

    await waitFor(() => expect(rpc.cancelOAuth).toHaveBeenCalledWith(oauthStart.sessionId))
    expect(openExternalLink).not.toHaveBeenCalled()
    expect(rpc.pollOAuth).not.toHaveBeenCalled()
    expect(createAntigravityRpc.mock.calls.at(-1)?.[1]()).toBe('default')
  })

  it('cancels a pending OAuth start under its original profile after a profile transition', async () => {
    const start = deferred<typeof oauthStart>()
    rpc.startOAuth.mockReturnValueOnce(start.promise)
    await renderAccounts()
    await screen.findByText('Antigravity accounts')

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Connect account' }))
    })
    await waitFor(() => expect(rpc.startOAuth).toHaveBeenCalledTimes(1))
    await act(async () => {
      $activeGatewayProfile.set('other')
    })
    await waitFor(() => expect(rpc.listAccounts).toHaveBeenCalledTimes(2))

    await act(async () => {
      start.resolve(oauthStart)
    })

    await waitFor(() => expect(rpc.cancelOAuth).toHaveBeenCalledWith(oauthStart.sessionId))
    expect(openExternalLink).not.toHaveBeenCalled()
    expect(rpc.pollOAuth).not.toHaveBeenCalled()
    expect(createAntigravityRpc.mock.calls.at(-1)?.[1]()).toBe('default')
  })

  it('clears rows and ignores stale account responses when the active profile changes', async () => {
    const first = deferred<{ accounts: AntigravityAccount[] }>()
    const second = deferred<{ accounts: AntigravityAccount[] }>()
    rpc.listAccounts.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)
    await renderAccounts()

    $activeGatewayProfile.set('other')
    await waitFor(() => expect(rpc.listAccounts).toHaveBeenCalledTimes(2))
    expect(createAntigravityRpc.mock.calls[0][1]()).toBe('other')
    expect(screen.queryByDisplayValue('2')).toBeNull()
    second.resolve(snapshot([account('2')]))
    await screen.findByDisplayValue('2')
    first.resolve(snapshot([account('1', { priority: 99 })]))

    await act(async () => undefined)
    expect(screen.queryByDisplayValue('99')).toBeNull()
  })
})
