import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register, SessionContextUsage, SessionRateLimit } from 'claude-code'

import type { UsageContext, UsageFile, UsageLimits, UsageTokens, UsageWindow } from '../types'
import {
  TERMINAL_BAR,
  TERMINAL_COLOR,
  TERMINAL_ICON,
  barLevel,
  formatPercent,
  pillGroups,
  pillSvg,
  summaryLine,
  terminalBar,
} from './pills'
import type { LimitPill, Pill, View } from './pills'

const limits = atom({ plugin: 'usage-band', key: 'limits' } as const, null)
const context = atom({ plugin: 'usage-band', key: 'context' } as const, null)
const costUsd = atom({ plugin: 'usage-band', key: 'costUsd' } as const, null)
const tokens = atom({ plugin: 'usage-band', key: 'tokens' } as const, null)
const fallback = atom({ plugin: 'usage-band', key: 'fallback' } as const, null)
const transcript = atom({ plugin: 'usage-band', key: 'transcript' } as const, null)
const isHidden = atom({ plugin: 'usage-band', key: 'isHidden' } as const, false)
const now = atom({ plugin: 'usage-band', key: 'now' } as const, 0)

const NODES = [
  'node',
  '/usr/local/bin/node',
  '/opt/homebrew/bin/node',
  'C:\\Program Files\\nodejs\\node.exe',
]
const TICK_MS = 30_000

type TokensReply = UsageFile & {
  input: number
  cacheWrite: number
  output: number
  cacheRead: number
  requests: number
}

function toWindow(list: readonly SessionRateLimit[], kind: string): UsageWindow | null {
  const found = list.find(one => one.kind === kind)
  return found ? { percentUsed: found.percentUsed, resetsAt: found.resetsAt ?? null } : null
}

function toLimits(list: readonly SessionRateLimit[]): UsageLimits | null {
  const fiveHour = toWindow(list, 'five_hour')
  const sevenDay = toWindow(list, 'seven_day')
  return fiveHour || sevenDay ? { fiveHour, sevenDay } : null
}

function toContext(c: SessionContextUsage): UsageContext {
  return { tokens: c.tokens ?? null, window: c.window, percent: c.percent ?? null }
}

// Modülün kendi değişkenleri: yeniden yüklemede sıfırlanır, sorun değil.
let node: string | null = null
let isCounting = false

async function measure($: EngineInterface): Promise<void> {
  const usage = await $.session.usage()
  const found = toLimits(usage.rateLimits)
  // Limitler ilk model yanıtıyla gelir; o zamana kadar boş kalır.
  if (found) await update($, limits, () => found)
  await update($, context, () => toContext(usage.context))
  if (usage.cost) {
    const usd = usage.cost.usd
    await update($, costUsd, () => usd)
  }
}

async function runScript($: EngineInterface, id: string): Promise<TokensReply | null> {
  const script = `${$.plugin.root}/scripts/tokens.mjs`
  const candidates = node ? [node, ...NODES.filter(n => n !== node)] : NODES
  for (const bin of candidates) {
    try {
      const ran = await $.process.run([bin, script, id], { timeoutMs: 20_000 })
      if (ran.exitCode !== 0) return null
      node = bin
      return JSON.parse(ran.stdout.trim()) as TokensReply
    } catch {
      // Bu node başlatılamadı; sıradakini dene.
    }
  }
  return null
}

async function count($: EngineInterface, force = false): Promise<void> {
  if (isCounting) return
  isCounting = true
  try {
    const last = await read($, transcript)
    if (!force && last) {
      const stat = await $.fs.stat(last.path).catch(() => null)
      if (stat && 'size' in stat && stat.size === last.size && stat.mtimeMs === last.mtimeMs) return
    }
    const reply = await runScript($, await $.session.id())
    if (reply === null) {
      // Betik çalışmadı: tur sonlarından toplanan yaklaşık değerler.
      const approx = await read($, fallback)
      if (approx) await update($, tokens, () => approx)
      return
    }
    const counted: UsageTokens = {
      input: reply.input,
      cacheWrite: reply.cacheWrite,
      output: reply.output,
      cacheRead: reply.cacheRead,
      requests: reply.requests,
      isApprox: false,
    }
    await update($, tokens, () => counted)
    await update($, transcript, () => ({ path: reply.path, size: reply.size, mtimeMs: reply.mtimeMs }))
  } finally {
    isCounting = false
  }
}

async function refresh($: EngineInterface, force = false): Promise<void> {
  const at = await $.clock.now()
  await update($, now, () => at)
  await Promise.all([measure($), count($, force)])
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    const started = await next(e)
    await $.command.register({
      name: 'kullanim',
      description: 'Kullanım şeridini yenile ve özetle (gizle / goster)',
    })
    void refresh($).catch(() => undefined)
    $.clock.every(TICK_MS, () => {
      void $.clock.now().then(at => update($, now, () => at))
    })
    return started
  })

  on('session.measure', async ($, e, next) => {
    const found = toLimits(e.rateLimits)
    if (found) await update($, limits, () => found)
    const ctx = toContext(e.context)
    await update($, context, () => ctx)
    if (e.cost) {
      const usd = e.cost.usd
      await update($, costUsd, () => usd)
    }
    return next(e)
  })

  on('turn.complete', async ($, e, next) => {
    const done = await next(e)
    const usage = e.usage
    if (usage) {
      await update($, fallback, prev => ({
        input: (prev?.input ?? 0) + usage.input_tokens,
        cacheWrite: (prev?.cacheWrite ?? 0) + usage.cache_creation_input_tokens,
        output: (prev?.output ?? 0) + usage.output_tokens,
        cacheRead: (prev?.cacheRead ?? 0) + usage.cache_read_input_tokens,
        requests: (prev?.requests ?? 0) + 1,
        isApprox: true,
      }))
    }
    void refresh($).catch(() => undefined)
    return done
  })

  on('command.run', { command: 'kullanim' }, async ($, e) => {
    const arg = e.args.trim().toLowerCase()
    if (arg === 'gizle') {
      await update($, isHidden, () => true)
      return { text: 'Kullanım şeridi gizlendi. Geri açmak için: /kullanim goster' }
    }
    if (arg === 'goster' || arg === 'göster') {
      await update($, isHidden, () => false)
    }
    await refresh($, true)
    const view: View = {
      now: await $.clock.now(),
      limits: await read($, limits),
      context: await read($, context),
      costUsd: await read($, costUsd),
      tokens: await read($, tokens),
    }
    return { text: summaryLine(view) }
  })

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    if (e.props.hasSurvey || (await read($, isHidden))) return next(e)

    const view: View = {
      now: (await read($, now)) || (await $.clock.now()),
      limits: await read($, limits),
      context: await read($, context),
      costUsd: await read($, costUsd),
      tokens: await read($, tokens),
    }
    const groups = pillGroups(view)
    if (groups.length === 0) return next(e)

    if (e.surface === 'desktop') {
      const { Box, Svg } = $.ui.resolve(e)
      return (
        <Box flexDirection="row" flexWrap="wrap" columnGap={2} rowGap={1}>
          {groups.map((group, g) => (
            <Box key={`g${g}`} flexDirection="row" flexWrap="wrap" columnGap={1} rowGap={1}>
              {group.map(pill => {
                const svg = pillSvg(pill)
                return (
                  <Svg
                    key={svg.key}
                    source={svg.source}
                    alt={svg.alt}
                    width={svg.width}
                    height={svg.height}
                    isInteractive={true}
                  />
                )
              })}
            </Box>
          ))}
        </Box>
      )
    }

    if (e.surface !== 'terminal') return next(e)
    const { Box, Text } = $.ui.resolve(e)

    const limitText = (pill: LimitPill) => {
      const bar = terminalBar(pill)
      const color = TERMINAL_COLOR[pill.tone]
      const barColor = TERMINAL_BAR[barLevel(pill.percent)]
      const cells = (bar.filled + bar.empty).split('')
      return (
        <Text>
          <Text color={color}>
            {TERMINAL_ICON[pill.icon]} {pill.label}{' '}
          </Text>
          {cells.map((cell, i) =>
            i === bar.markAt ? (
              <Text color={color}>│</Text>
            ) : (
              <Text color={barColor} dimColor={cell === '░'}>
                {cell}
              </Text>
            ),
          )}
          <Text color={color} bold>
            {' '}
            {formatPercent(pill.percent)}
          </Text>
          <Text dimColor> · </Text>
          <Text color={color}>⏳ {pill.left}</Text>
        </Text>
      )
    }

    const valueText = (pill: Pill) =>
      pill.kind === 'limit' ? (
        limitText(pill)
      ) : (
        <Text color={TERMINAL_COLOR[pill.tone]}>
          {TERMINAL_ICON[pill.icon]} <Text bold>{pill.text}</Text>
        </Text>
      )

    return (
      <Box flexDirection="row" flexWrap="wrap" columnGap={3}>
        {groups.map((group, g) => (
          <Box key={`g${g}`} flexDirection="row" flexWrap="wrap" columnGap={2}>
            {group.map(pill => valueText(pill))}
          </Box>
        ))}
      </Box>
    )
  })
}
