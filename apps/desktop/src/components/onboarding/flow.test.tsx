import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { makeOAuthProvider } from '@/test/oauth-provider'

import { FlowPanel } from './flow'

describe('FlowPanel browser polling', () => {
  it('shows the existing browser waiting actions without device-code or terminal instructions', () => {
    const provider = makeOAuthProvider('cursor', 'Cursor')
    provider.flow = 'browser_poll'
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <FlowPanel
          ctx={{ profile: 'selected', requestGateway: vi.fn() }}
          flow={{
            provider,
            start: {
              auth_url: 'https://cursor.example/sign-in',
              expires_in: 600,
              flow: 'browser_poll',
              poll_interval: 2,
              session_id: 'session'
            },
            status: 'browser_polling'
          }}
          leaving={false}
          onBegin={vi.fn()}
        />
      </I18nProvider>
    )

    expect(screen.getByText(/opened Cursor in your browser/i)).toBeTruthy()
    expect(screen.getByText('Waiting for you to authorize...')).toBeTruthy()
    expect(screen.getByRole('link', { name: 'Re-open sign-in page' }).getAttribute('href')).toBe(
      'https://cursor.example/sign-in'
    )
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeTruthy()
    expect(screen.queryByText(/device code|one-time|hermes auth add cursor/i)).toBeNull()
  })
})
