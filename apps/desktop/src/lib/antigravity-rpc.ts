export const ANTIGRAVITY_RPC_METHODS = {
  cancel: 'antigravity.oauth.cancel',
  enabled: 'antigravity.accounts.enabled',
  list: 'antigravity.accounts.list',
  poll: 'antigravity.oauth.poll',
  priority: 'antigravity.accounts.priority',
  remove: 'antigravity.accounts.remove',
  start: 'antigravity.oauth.start'
} as const

const MAX_PROJECT_ID_LENGTH = 4096
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
const ACCOUNT_ID_PATTERN = new RegExp(`^acct_${UUID_PATTERN.source.slice(1, -1)}$`)

export interface AntigravityAccount {
  enabled: boolean
  id: string
  priority: number
}

export interface AntigravityAccountsSnapshot {
  accounts: AntigravityAccount[]
}

export interface AntigravityOAuthStart {
  authUrl: string
  expiresAt: number
  flow: string
  pollIntervalMs: number
  sessionId: string
  status: string
}

export interface AntigravityOAuthStatus {
  status: string
}

export interface AntigravityRequestOptions {
  signal?: AbortSignal
}

export type AntigravityGatewayRequest = (
  method: string,
  params?: Record<string, unknown>,
  timeoutMs?: number,
  signal?: AbortSignal
) => Promise<unknown>

export interface AntigravityRpc {
  cancelOAuth(sessionId: string, options?: AntigravityRequestOptions): Promise<AntigravityOAuthStatus>
  listAccounts(options?: AntigravityRequestOptions): Promise<AntigravityAccountsSnapshot>
  pollOAuth(sessionId: string, options?: AntigravityRequestOptions): Promise<AntigravityOAuthStatus>
  removeAccount(accountId: string, options?: AntigravityRequestOptions): Promise<AntigravityAccountsSnapshot>
  setAccountEnabled(
    accountId: string,
    enabled: boolean,
    options?: AntigravityRequestOptions
  ): Promise<AntigravityAccountsSnapshot>
  setAccountPriority(
    accountId: string,
    priority: number,
    options?: AntigravityRequestOptions
  ): Promise<AntigravityAccountsSnapshot>
  startOAuth(projectId: string, options?: AntigravityRequestOptions): Promise<AntigravityOAuthStart>
}

interface AccountsEnvelope {
  accounts: unknown
}

interface SnapshotEnvelope {
  snapshot: unknown
}

interface OAuthStartEnvelope {
  auth_url: unknown
  expires_at: unknown
  flow: unknown
  poll_interval_ms: unknown
  session_id: unknown
  status: unknown
}

interface OAuthStatusEnvelope {
  status: unknown
}

const invalidRequest = (): Error => new Error('Invalid Antigravity request.')
const invalidResponse = (): Error => new Error('Invalid Antigravity response.')

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null) {
    return false
  }

  const prototype = Object.getPrototypeOf(value)

  return prototype === Object.prototype || prototype === null
}

function isSafeText(value: unknown, maximumLength = 4096): value is string {
  return (
    typeof value === 'string' &&
    value.length > 0 &&
    value.length <= maximumLength &&
    !Array.from(value).some(character => {
      const codePoint = character.codePointAt(0) ?? 0

      return codePoint < 32 || codePoint === 127
    })
  )
}

function isUuid(value: unknown): value is string {
  return typeof value === 'string' && UUID_PATTERN.test(value)
}

function isAccountId(value: unknown): value is string {
  return typeof value === 'string' && ACCOUNT_ID_PATTERN.test(value)
}

function isPriority(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 1
}

function requiredProfile(getActiveProfile: () => string): string {
  const profile = getActiveProfile()

  if (!isSafeText(profile, 255) || profile.trim().length === 0) {
    throw invalidRequest()
  }

  return profile
}

function parseAccount(value: unknown): AntigravityAccount {
  if (
    !isPlainRecord(value) ||
    !isAccountId(value.id) ||
    typeof value.enabled !== 'boolean' ||
    !isPriority(value.priority)
  ) {
    throw invalidResponse()
  }

  return { enabled: value.enabled, id: value.id, priority: value.priority }
}

function parseSnapshot(value: unknown): AntigravityAccountsSnapshot {
  if (!isPlainRecord(value) || !Array.isArray(value.accounts)) {
    throw invalidResponse()
  }

  return { accounts: value.accounts.map(parseAccount) }
}

function parseAccountsEnvelope(value: unknown): AntigravityAccountsSnapshot {
  if (!isPlainRecord(value)) {
    throw invalidResponse()
  }

  const envelope: AccountsEnvelope = { accounts: value.accounts }

  return parseSnapshot(envelope.accounts)
}

function parseSnapshotEnvelope(value: unknown): AntigravityAccountsSnapshot {
  if (!isPlainRecord(value)) {
    throw invalidResponse()
  }

  const envelope: SnapshotEnvelope = { snapshot: value.snapshot }

  return parseSnapshot(envelope.snapshot)
}

function parseOAuthStart(value: unknown): AntigravityOAuthStart {
  if (!isPlainRecord(value)) {
    throw invalidResponse()
  }

  const envelope: OAuthStartEnvelope = {
    auth_url: value.auth_url,
    expires_at: value.expires_at,
    flow: value.flow,
    poll_interval_ms: value.poll_interval_ms,
    session_id: value.session_id,
    status: value.status
  }

  if (
    !isSafeText(envelope.auth_url) ||
    typeof envelope.expires_at !== 'number' ||
    !Number.isSafeInteger(envelope.expires_at) ||
    !isSafeText(envelope.flow, 128) ||
    !isPriority(envelope.poll_interval_ms) ||
    !isUuid(envelope.session_id) ||
    !isSafeText(envelope.status, 128)
  ) {
    throw invalidResponse()
  }

  return {
    authUrl: envelope.auth_url,
    expiresAt: envelope.expires_at,
    flow: envelope.flow,
    pollIntervalMs: envelope.poll_interval_ms,
    sessionId: envelope.session_id,
    status: envelope.status
  }
}

function parseOAuthStatus(value: unknown): AntigravityOAuthStatus {
  if (!isPlainRecord(value)) {
    throw invalidResponse()
  }

  const envelope: OAuthStatusEnvelope = { status: value.status }

  if (!isSafeText(envelope.status, 128)) {
    throw invalidResponse()
  }

  return { status: envelope.status }
}

function requestParams(getActiveProfile: () => string, params: Record<string, unknown> = {}): Record<string, unknown> {
  return { ...params, profile: requiredProfile(getActiveProfile) }
}

/**
 * Framework-agnostic Antigravity account/OAuth RPC actions. The caller supplies
 * the gateway request function and active-profile reader so this module stays
 * independent of React while every request remains profile-scoped.
 */
export function createAntigravityRpc(
  requestGateway: AntigravityGatewayRequest,
  getActiveProfile: () => string
): AntigravityRpc {
  const request = async (
    method: string,
    params?: Record<string, unknown>,
    options?: AntigravityRequestOptions
  ): Promise<unknown> => {
    const scopedParams = requestParams(getActiveProfile, params)

    try {
      return await requestGateway(method, scopedParams, undefined, options?.signal)
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw error
      }

      throw new Error('Antigravity request failed.')
    }
  }

  return {
    cancelOAuth: async (sessionId, options) => {
      if (!isUuid(sessionId)) {
        throw invalidRequest()
      }

      return parseOAuthStatus(await request(ANTIGRAVITY_RPC_METHODS.cancel, { session_id: sessionId }, options))
    },
    listAccounts: async options =>
      parseAccountsEnvelope(await request(ANTIGRAVITY_RPC_METHODS.list, undefined, options)),
    pollOAuth: async (sessionId, options) => {
      if (!isUuid(sessionId)) {
        throw invalidRequest()
      }

      return parseOAuthStatus(await request(ANTIGRAVITY_RPC_METHODS.poll, { session_id: sessionId }, options))
    },
    removeAccount: async (accountId, options) => {
      if (!isAccountId(accountId)) {
        throw invalidRequest()
      }

      return parseSnapshotEnvelope(await request(ANTIGRAVITY_RPC_METHODS.remove, { account_id: accountId }, options))
    },
    setAccountEnabled: async (accountId, enabled, options) => {
      if (!isAccountId(accountId) || typeof enabled !== 'boolean') {
        throw invalidRequest()
      }

      return parseSnapshotEnvelope(
        await request(ANTIGRAVITY_RPC_METHODS.enabled, { account_id: accountId, enabled }, options)
      )
    },
    setAccountPriority: async (accountId, priority, options) => {
      if (!isAccountId(accountId) || !isPriority(priority)) {
        throw invalidRequest()
      }

      return parseSnapshotEnvelope(
        await request(ANTIGRAVITY_RPC_METHODS.priority, { account_id: accountId, priority }, options)
      )
    },
    startOAuth: async (projectId, options) => {
      if (!isSafeText(projectId, MAX_PROJECT_ID_LENGTH)) {
        throw invalidRequest()
      }

      return parseOAuthStart(await request(ANTIGRAVITY_RPC_METHODS.start, { project_id: projectId }, options))
    }
  }
}
