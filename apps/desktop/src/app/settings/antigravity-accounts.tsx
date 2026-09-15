import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import {
  type AntigravityAccount,
  type AntigravityAccountsSnapshot,
  type AntigravityRpc,
  createAntigravityRpc
} from '@/lib/antigravity-rpc'
import { openExternalLink } from '@/lib/external-link'
import { Loader2, Plus, RefreshCw, Trash2 } from '@/lib/icons'
import { confirm } from '@/store/confirm'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'

import { EmptyState, ListRow } from './primitives'

const MAX_PRIORITY = 999_999
const COMPLETE_OAUTH_STATUSES = new Set(['approved', 'complete', 'completed', 'success', 'succeeded'])

interface PendingOAuthStart {
  controller: AbortController
  profile: string
}

interface OAuthFlow {
  controller: AbortController
  profile: string
  sessionId: string
  timer: ReturnType<typeof setInterval>
}

function isAbort(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError'
}

interface AntigravityAccountsProps {
  embedded?: boolean
  onConfigSaved?: () => void
}

export function AntigravityAccounts({ embedded, onConfigSaved }: AntigravityAccountsProps) {
  const profile = useStore($activeGatewayProfile)

  return <AntigravityAccountsForProfile embedded={embedded} key={profile} onConfigSaved={onConfigSaved} />
}

function AntigravityAccountsForProfile({ embedded, onConfigSaved }: AntigravityAccountsProps) {
  const { t } = useI18n()
  const copy = t.settings.providers.antigravity
  const { requestGateway } = useGatewayRequest()
  const rpc = useMemo(() => createAntigravityRpc(requestGateway, () => $activeGatewayProfile.get()), [requestGateway])
  const [accounts, setAccounts] = useState<AntigravityAccount[] | null>(null)
  const [error, setError] = useState(false)

  const [priorityDrafts, setPriorityDrafts] = useState<Record<string, string>>({})
  const [priorityErrors, setPriorityErrors] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState<null | string>(null)
  const [connecting, setConnecting] = useState(false)
  const generation = useRef(0)
  const oauth = useRef<OAuthFlow | null>(null)
  const pendingOAuthStart = useRef<PendingOAuthStart | null>(null)

  const applySnapshot = useCallback(
    (snapshot: AntigravityAccountsSnapshot, expectedProfile: string, expectedGeneration: number) => {
      if (expectedProfile !== $activeGatewayProfile.get() || expectedGeneration !== generation.current) {
        return false
      }

      setAccounts(snapshot.accounts)
      setPriorityDrafts(Object.fromEntries(snapshot.accounts.map(account => [account.id, String(account.priority)])))
      setPriorityErrors({})
      setError(false)

      return true
    },
    []
  )

  const refresh = useCallback(
    async (controller = new AbortController()) => {
      const expectedProfile = $activeGatewayProfile.get()
      const expectedGeneration = ++generation.current
      setError(false)

      try {
        const snapshot = await rpc.listAccounts({ signal: controller.signal })

        return applySnapshot(snapshot, expectedProfile, expectedGeneration)
      } catch (reason) {
        if (
          !isAbort(reason) &&
          expectedProfile === $activeGatewayProfile.get() &&
          expectedGeneration === generation.current
        ) {
          setError(true)
        }

        return false
      }
    },
    [applySnapshot, rpc]
  )

  const cancelOAuth = useCallback(
    (flow = oauth.current, updateState = true) => {
      const pending = pendingOAuthStart.current

      if (pending) {
        pendingOAuthStart.current = null
        pending.controller.abort()
      }

      if (!flow) {
        if (updateState && pending) {
          setConnecting(false)
        }

        return
      }

      if (oauth.current === flow) {
        oauth.current = null
      }

      clearInterval(flow.timer)
      flow.controller.abort()
      const scopedRpc = createAntigravityRpc(requestGateway, () => flow.profile)
      void scopedRpc.cancelOAuth(flow.sessionId).catch(() => undefined)

      if (updateState) {
        setConnecting(false)
      }
    },
    [requestGateway]
  )

  useEffect(() => {
    const controller = new AbortController()
    setAccounts(null)
    setPriorityDrafts({})
    setPriorityErrors({})
    void refresh(controller)

    return () => {
      controller.abort()
      cancelOAuth(undefined, false)
    }
  }, [cancelOAuth, refresh])

  const runMutation = useCallback(
    async (
      key: string,
      operation: (client: AntigravityRpc, signal: AbortSignal) => Promise<AntigravityAccountsSnapshot>
    ) => {
      if (busy !== null) {
        return
      }

      const expectedProfile = $activeGatewayProfile.get()
      const expectedGeneration = generation.current
      const controller = new AbortController()
      setBusy(key)

      try {
        const snapshot = await operation(rpc, controller.signal)

        return applySnapshot(snapshot, expectedProfile, expectedGeneration)
      } catch (reason) {
        if (!isAbort(reason)) {
          notifyError(reason, copy.mutationFailed)
        }

        return false
      } finally {
        setBusy(current => (current === key ? null : current))
      }
    },
    [applySnapshot, busy, copy.mutationFailed, rpc]
  )

  const changeEnabled = (account: AntigravityAccount, enabled: boolean) =>
    runMutation(`enabled:${account.id}`, (client, signal) => client.setAccountEnabled(account.id, enabled, { signal }))

  const savePriority = (account: AntigravityAccount) => {
    const draft = priorityDrafts[account.id] ?? String(account.priority)
    const value = Number(draft)

    if (!Number.isSafeInteger(value) || value < 1 || value > MAX_PRIORITY) {
      setPriorityErrors(current => ({ ...current, [account.id]: copy.priorityInvalid }))

      return
    }

    setPriorityErrors(current => {
      const { [account.id]: _, ...rest } = current

      return rest
    })
    void runMutation(`priority:${account.id}`, (client, signal) =>
      client.setAccountPriority(account.id, value, { signal })
    )
  }

  const remove = async (account: AntigravityAccount) => {
    if (
      busy !== null ||
      !(await confirm({ confirmLabel: t.common.remove, destructive: true, title: copy.removeConfirm }))
    ) {
      return
    }

    if (await runMutation(`remove:${account.id}`, (client, signal) => client.removeAccount(account.id, { signal }))) {
      onConfigSaved?.()
    }
  }

  const poll = useCallback(
    async (flow: OAuthFlow) => {
      try {
        const result = await rpc.pollOAuth(flow.sessionId, { signal: flow.controller.signal })

        if (oauth.current !== flow || flow.profile !== $activeGatewayProfile.get()) {
          return
        }

        if (COMPLETE_OAUTH_STATUSES.has(result.status.toLowerCase())) {
          clearInterval(flow.timer)
          oauth.current = null
          setConnecting(false)
          onConfigSaved?.()
          await refresh()
        }
      } catch (reason) {
        if (!isAbort(reason) && oauth.current === flow) {
          cancelOAuth(flow)
          notifyError(reason, copy.connectFailed)
        }
      }
    },
    [cancelOAuth, copy.connectFailed, onConfigSaved, refresh, rpc]
  )

  const connect = async () => {
    if (connecting || pendingOAuthStart.current) {
      return
    }

    const pending: PendingOAuthStart = {
      controller: new AbortController(),
      profile: $activeGatewayProfile.get()
    }

    pendingOAuthStart.current = pending
    setConnecting(true)

    try {
      const start = await rpc.startOAuth({ signal: pending.controller.signal })

      if (pendingOAuthStart.current !== pending || pending.profile !== $activeGatewayProfile.get()) {
        const scopedRpc = createAntigravityRpc(requestGateway, () => pending.profile)
        void scopedRpc.cancelOAuth(start.sessionId).catch(() => undefined)

        return
      }

      pendingOAuthStart.current = null
      // startOAuth validates the authorization URL before this presentation layer receives it.
      openExternalLink(start.authUrl)

      const flow: OAuthFlow = {
        controller: pending.controller,
        profile: pending.profile,
        sessionId: start.sessionId,
        timer: setInterval(() => void poll(flow), start.pollIntervalMs)
      }

      oauth.current = flow
    } catch (reason) {
      if (pendingOAuthStart.current === pending) {
        pendingOAuthStart.current = null

        if (!isAbort(reason)) {
          notifyError(reason, copy.connectFailed)
        }

        setConnecting(false)
      }
    }
  }

  return (
    <div className="grid gap-3">
      <div className="flex items-center justify-between gap-2">
        {!embedded && <h3>{copy.title}</h3>}
        <Button
          disabled={accounts === null || busy !== null || connecting}
          onClick={() => void refresh()}
          size="sm"
          type="button"
          variant="ghost"
        >
          <RefreshCw className="size-3.5" />
          {t.common.refresh}
        </Button>
      </div>
      <div className="grid gap-3">
        <p className="text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {copy.description}
        </p>
        <div className="flex flex-wrap items-end gap-2">
          {connecting ? (
            <Button onClick={() => cancelOAuth()} type="button" variant="outline">
              {t.common.cancel}
            </Button>
          ) : (
            <Button onClick={() => void connect()} type="button">
              <Plus className="size-4" />
              {copy.connect}
            </Button>
          )}
        </div>
        {error ? (
          <div className="grid justify-items-center gap-3 py-5 text-center">
            <EmptyState description={copy.loadFailed} title={t.common.error} />
            <Button onClick={() => void refresh()} size="sm" type="button" variant="outline">
              {t.common.retry}
            </Button>
          </div>
        ) : accounts === null ? (
          <div aria-live="polite" className="py-5 text-center text-sm text-muted-foreground" role="status">
            <Loader2 className="mr-2 inline size-4 animate-spin" />
            {t.common.loading}
          </div>
        ) : accounts.length === 0 ? (
          <EmptyState description={copy.empty} title={copy.emptyTitle} />
        ) : (
          <div className="grid gap-2">
            {accounts.map((account, index) => {
              const label = account.email || copy.accountLabel(String(index + 1))
              const accessibleName = account.email || String(index + 1)
              const isBusy = busy?.endsWith(account.id) ?? false

              return (
                <ListRow
                  action={
                    <div className="flex flex-wrap items-center justify-end gap-2">
                      <Switch
                        aria-label={copy.enableAccount(accessibleName)}
                        checked={account.enabled}
                        disabled={isBusy}
                        onCheckedChange={enabled => void changeEnabled(account, enabled)}
                      />
                      <label className="grid gap-1 text-xs text-muted-foreground">
                        {copy.priority}
                        <Input
                          aria-describedby={priorityErrors[account.id] ? `priority-error-${account.id}` : undefined}
                          aria-label={copy.priorityFor(accessibleName)}
                          disabled={isBusy}
                          inputMode="numeric"
                          max={MAX_PRIORITY}
                          min={1}
                          onChange={event =>
                            setPriorityDrafts(current => ({ ...current, [account.id]: event.target.value }))
                          }
                          type="number"
                          value={priorityDrafts[account.id] ?? String(account.priority)}
                        />
                      </label>
                      <Button
                        disabled={isBusy}
                        onClick={() => savePriority(account)}
                        size="sm"
                        type="button"
                        variant="outline"
                      >
                        {copy.savePriority}
                      </Button>
                      <Button
                        aria-label={copy.removeAccount(accessibleName)}
                        disabled={isBusy}
                        onClick={() => void remove(account)}
                        size="icon-sm"
                        type="button"
                        variant="ghost"
                      >
                        {busy === `remove:${account.id}` ? (
                          <Loader2 className="size-4 animate-spin" />
                        ) : (
                          <Trash2 className="size-4" />
                        )}
                      </Button>
                    </div>
                  }
                  below={
                    priorityErrors[account.id] ? (
                      <p className="mt-1 text-xs text-destructive" id={`priority-error-${account.id}`} role="alert">
                        {priorityErrors[account.id]}
                      </p>
                    ) : undefined
                  }
                  description={
                    account.status === 'verification_required'
                      ? copy.verificationRequired
                      : account.enabled
                        ? copy.enabled
                        : copy.disabled
                  }
                  key={account.id}
                  title={<span>{label}</span>}
                />
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
