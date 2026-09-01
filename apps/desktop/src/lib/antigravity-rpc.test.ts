import { describe, expect, it, vi } from 'vitest'

import { ANTIGRAVITY_RPC_METHODS, type AntigravityGatewayRequest, createAntigravityRpc } from './antigravity-rpc'

const ACCOUNT_ID = 'acct_123e4567-e89b-12d3-a456-426614174000'
const SESSION_ID = '123e4567-e89b-12d3-a456-426614174000'

const snapshot = {
  accounts: [{ enabled: true, id: ACCOUNT_ID, priority: 1 }]
}

function requestStub(): AntigravityGatewayRequest {
  const responseForMethod = (method: string): unknown => {
    switch (method) {
      case ANTIGRAVITY_RPC_METHODS.list:
        return { accounts: snapshot }

      case ANTIGRAVITY_RPC_METHODS.enabled:

      case ANTIGRAVITY_RPC_METHODS.priority:

      case ANTIGRAVITY_RPC_METHODS.remove:
        return { snapshot }

      case ANTIGRAVITY_RPC_METHODS.start:
        return {
          auth_url: 'https://accounts.example.test/authorize',
          expires_at: 123,
          flow: 'browser_poll',
          poll_interval_ms: 1000,
          session_id: SESSION_ID,
          status: 'pending'
        }

      case ANTIGRAVITY_RPC_METHODS.poll:

      case ANTIGRAVITY_RPC_METHODS.cancel:
        return { status: 'cancelled' }

      default:
        throw new Error('unexpected method')
    }
  }

  return vi.fn(async (method: string): Promise<unknown> => responseForMethod(method))
}

describe('createAntigravityRpc', () => {
  it('forwards each exact RPC method, active profile, signal, and typed result', async () => {
    const request = requestStub()
    const signal = new AbortController().signal
    const api = createAntigravityRpc(request, () => 'alpha')

    await expect(api.listAccounts({ signal })).resolves.toEqual(snapshot)
    await expect(api.setAccountEnabled(ACCOUNT_ID, false, { signal })).resolves.toEqual(snapshot)
    await expect(api.setAccountPriority(ACCOUNT_ID, 2, { signal })).resolves.toEqual(snapshot)
    await expect(api.removeAccount(ACCOUNT_ID, { signal })).resolves.toEqual(snapshot)
    await expect(api.startOAuth('project-a', { signal })).resolves.toMatchObject({
      sessionId: SESSION_ID,
      status: 'pending'
    })
    await expect(api.pollOAuth(SESSION_ID, { signal })).resolves.toEqual({ status: 'cancelled' })
    await expect(api.cancelOAuth(SESSION_ID, { signal })).resolves.toEqual({ status: 'cancelled' })

    expect(request).toHaveBeenNthCalledWith(1, 'antigravity.accounts.list', { profile: 'alpha' }, undefined, signal)
    expect(request).toHaveBeenNthCalledWith(
      2,
      'antigravity.accounts.enabled',
      { account_id: ACCOUNT_ID, enabled: false, profile: 'alpha' },
      undefined,
      signal
    )
    expect(request).toHaveBeenNthCalledWith(
      3,
      'antigravity.accounts.priority',
      { account_id: ACCOUNT_ID, priority: 2, profile: 'alpha' },
      undefined,
      signal
    )
    expect(request).toHaveBeenNthCalledWith(
      4,
      'antigravity.accounts.remove',
      { account_id: ACCOUNT_ID, profile: 'alpha' },
      undefined,
      signal
    )
    expect(request).toHaveBeenNthCalledWith(
      5,
      'antigravity.oauth.start',
      { profile: 'alpha', project_id: 'project-a' },
      undefined,
      signal
    )
    expect(request).toHaveBeenNthCalledWith(
      6,
      'antigravity.oauth.poll',
      { profile: 'alpha', session_id: SESSION_ID },
      undefined,
      signal
    )
    expect(request).toHaveBeenNthCalledWith(
      7,
      'antigravity.oauth.cancel',
      { profile: 'alpha', session_id: SESSION_ID },
      undefined,
      signal
    )
  })

  it('reads the active profile for every request instead of holding stale state', async () => {
    const request = requestStub()
    let profile = 'alpha'
    const api = createAntigravityRpc(request, () => profile)

    await api.listAccounts()
    profile = 'beta'
    await api.listAccounts()

    expect(request).toHaveBeenNthCalledWith(1, 'antigravity.accounts.list', { profile: 'alpha' }, undefined, undefined)
    expect(request).toHaveBeenNthCalledWith(2, 'antigravity.accounts.list', { profile: 'beta' }, undefined, undefined)
  })

  it.each([
    ['account id with a bad prefix', (api: ReturnType<typeof createAntigravityRpc>) => api.removeAccount('account_x')],
    [
      'account id with uppercase UUID',
      (api: ReturnType<typeof createAntigravityRpc>) => api.removeAccount('acct_123E4567-e89b-12d3-a456-426614174000')
    ],
    [
      'boolean-like enabled value',
      (api: ReturnType<typeof createAntigravityRpc>) => Reflect.apply(api.setAccountEnabled, api, [ACCOUNT_ID, 1])
    ],
    [
      'unsafe priority',
      (api: ReturnType<typeof createAntigravityRpc>) => api.setAccountPriority(ACCOUNT_ID, Number.MAX_SAFE_INTEGER + 1)
    ],
    ['fractional priority', (api: ReturnType<typeof createAntigravityRpc>) => api.setAccountPriority(ACCOUNT_ID, 1.5)],
    ['empty project id', (api: ReturnType<typeof createAntigravityRpc>) => api.startOAuth('')],
    ['overlong project id', (api: ReturnType<typeof createAntigravityRpc>) => api.startOAuth('x'.repeat(4097))],
    ['control-character project id', (api: ReturnType<typeof createAntigravityRpc>) => api.startOAuth('bad\nproject')],
    [
      'malformed OAuth poll session id',
      (api: ReturnType<typeof createAntigravityRpc>) => api.pollOAuth('not-a-session')
    ],
    [
      'malformed OAuth cancel session id',
      (api: ReturnType<typeof createAntigravityRpc>) => api.cancelOAuth('not-a-session')
    ]
  ])('rejects %s before invoking the request function', async (_name, invoke) => {
    const request = requestStub()
    const api = createAntigravityRpc(request, () => 'alpha')

    await expect(invoke(api)).rejects.toThrow('Invalid Antigravity request.')
    expect(request).not.toHaveBeenCalled()
  })

  it('rejects hostile values and malformed backend envelopes without trusting them', async () => {
    const hostile = Object.create({ accounts: snapshot.accounts })
    const request: AntigravityGatewayRequest = vi.fn(async () => hostile)
    const api = createAntigravityRpc(request, () => 'alpha')

    await expect(api.listAccounts()).rejects.toThrow('Invalid Antigravity response.')

    const malformed: AntigravityGatewayRequest = vi.fn(async () => ({
      accounts: { accounts: [{ enabled: 1, id: ACCOUNT_ID, priority: 1 }] }
    }))

    await expect(createAntigravityRpc(malformed, () => 'alpha').listAccounts()).rejects.toThrow(
      'Invalid Antigravity response.'
    )

    const malformedOAuth: AntigravityGatewayRequest = vi.fn(async () => ({
      auth_url: 'https://accounts.example.test/authorize',
      expires_at: 123,
      flow: 'browser_poll',
      poll_interval_ms: 1000,
      session_id: SESSION_ID,
      status: 1
    }))

    await expect(createAntigravityRpc(malformedOAuth, () => 'alpha').startOAuth('project-a')).rejects.toThrow(
      'Invalid Antigravity response.'
    )
  })

  it('converts gateway failures to a fixed safe error while preserving cancellation', async () => {
    const failing: AntigravityGatewayRequest = vi.fn(async () => {
      throw new Error('server payload: private-project-value')
    })

    await expect(createAntigravityRpc(failing, () => 'alpha').listAccounts()).rejects.toThrow(
      'Antigravity request failed.'
    )

    const aborted: AntigravityGatewayRequest = vi.fn(async () => {
      const error = new Error('aborted')

      error.name = 'AbortError'
      throw error
    })

    await expect(createAntigravityRpc(aborted, () => 'alpha').listAccounts()).rejects.toMatchObject({
      name: 'AbortError'
    })
  })
})
