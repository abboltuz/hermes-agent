import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ConfirmHost } from '@/components/confirm-host'
import type { AntigravityAccount, AntigravityRpc } from '@/lib/antigravity-rpc'
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
  let resolve!: (value: T) => void
  const promise = new Promise<T>(innerResolve => {
    resolve = innerResolve
  })

  return { promise, resolve }
}

async function renderAccounts() {
  const { AntigravityAccounts } = await import('./antigravity-accounts')

  return render(
    <>
      <AntigravityAccounts />
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
    expect(screen.getByRole('switch', { name: /enable antigravity account/i }).getAttribute('aria-checked')).toBe('false')
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

  it('starts OAuth only through the typed client and cancels its poll lifecycle', async () => {
    await renderAccounts()
    await screen.findByText('Antigravity accounts')

    fireEvent.change(screen.getByLabelText('Project ID'), { target: { value: 'safe-project' } })
    vi.useFakeTimers()
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Connect account' }))
    })

    expect(rpc.startOAuth).toHaveBeenCalledWith('safe-project', expect.anything())
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

  it('cancels OAuth when the settings surface unmounts', async () => {
    const view = await renderAccounts()
    await screen.findByText('Antigravity accounts')

    fireEvent.click(screen.getByRole('button', { name: 'Connect account' }))
    await waitFor(() => expect(rpc.startOAuth).toHaveBeenCalledTimes(1))
    view.unmount()

    expect(rpc.cancelOAuth).toHaveBeenCalledWith(oauthStart.sessionId)
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
