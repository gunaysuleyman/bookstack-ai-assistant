export type UsageWindow = { percentUsed: number; resetsAt: string | null }

export type UsageLimits = { fiveHour: UsageWindow | null; sevenDay: UsageWindow | null }

export type UsageContext = { tokens: number | null; window: number; percent: number | null }

export type UsageTokens = {
  input: number
  cacheWrite: number
  output: number
  cacheRead: number
  requests: number
  isApprox: boolean
}

export type UsageFile = { path: string; size: number; mtimeMs: number }

declare module 'claude-code' {
  interface PluginState {
    'usage-band': {
      limits: UsageLimits | null
      context: UsageContext | null
      costUsd: number | null
      tokens: UsageTokens | null
      fallback: UsageTokens | null
      transcript: UsageFile | null
      isHidden: boolean
      now: number
    }
  }
}
