// Şeridin çizimden bağımsız kısmı: biçimlendirme, haplar ve SVG metni.
// Saf fonksiyonlar; hem hook modülü hem önizleme sayfası kullanır.
import type { UsageContext, UsageLimits, UsageTokens, UsageWindow } from '../types'

export type View = {
  now: number
  limits: UsageLimits | null
  context: UsageContext | null
  costUsd: number | null
  tokens: UsageTokens | null
}

export type Tone = 'teal' | 'purple' | 'red' | 'green' | 'blue' | 'gold'
export type IconName = 'gauge' | 'calendar' | 'up' | 'down' | 'layers' | 'dollar'

export type LimitPill = {
  kind: 'limit'
  key: string
  tone: Tone
  icon: IconName
  label: string
  percent: number
  elapsed: number | null
  left: string
  title: string
}

export type ValuePill = {
  kind: 'value'
  key: string
  tone: Tone
  icon: IconName
  text: string
  title: string
}

export type Pill = LimitPill | ValuePill

export const FIVE_HOURS = 5 * 3600 * 1000
export const SEVEN_DAYS = 7 * 24 * 3600 * 1000

export function formatTokens(n: number): string {
  if (n < 1000) return String(Math.round(n))
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`
  return `${(n / 1_000_000).toFixed(2)}M`
}

export function formatLeft(ms: number): string {
  const minutes = Math.max(0, Math.floor(ms / 60_000))
  const d = Math.floor(minutes / 1440)
  const h = Math.floor((minutes % 1440) / 60)
  const m = minutes % 60
  if (d > 0) return `${d}d ${h}h`
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

export function formatPercent(p: number): string {
  return `${Math.round(p)}%`
}

function clock(iso: string): string {
  const at = new Date(iso)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(at.getDate())}.${pad(at.getMonth() + 1)} ${pad(at.getHours())}:${pad(at.getMinutes())}`
}

export function barLevel(percent: number): 'ok' | 'warn' | 'high' {
  if (percent >= 90) return 'high'
  if (percent >= 70) return 'warn'
  return 'ok'
}

function limitPill(
  key: string,
  label: string,
  name: string,
  win: UsageWindow,
  span: number,
  tone: Tone,
  icon: IconName,
  now: number,
): LimitPill {
  const leftMs = win.resetsAt === null ? null : Date.parse(win.resetsAt) - now
  const elapsed =
    leftMs === null || Number.isNaN(leftMs) ? null : Math.min(1, Math.max(0, 1 - leftMs / span))
  const left = leftMs === null || Number.isNaN(leftMs) ? '—' : formatLeft(leftMs)
  const lines = [`${name}: ${formatPercent(win.percentUsed)} kullanıldı`]
  if (win.resetsAt !== null && leftMs !== null && !Number.isNaN(leftMs)) {
    lines.push(`Sıfırlanma: ${left} sonra (${clock(win.resetsAt)})`)
  }
  if (elapsed !== null) lines.push(`Pencerenin ${formatPercent(elapsed * 100)}'i geçti`)
  return {
    kind: 'limit',
    key,
    tone,
    icon,
    label,
    percent: win.percentUsed,
    elapsed,
    left,
    title: lines.join('\n'),
  }
}

export function pillGroups(view: View): Pill[][] {
  const groups: Pill[][] = []
  const { limits, context, tokens, costUsd, now } = view

  const limitGroup: Pill[] = []
  if (limits?.fiveHour) {
    limitGroup.push(limitPill('5h', '5h', '5 saatlik limit', limits.fiveHour, FIVE_HOURS, 'teal', 'gauge', now))
  }
  if (limits?.sevenDay) {
    limitGroup.push(limitPill('7d', '7d', '7 günlük limit', limits.sevenDay, SEVEN_DAYS, 'purple', 'calendar', now))
  }
  if (limitGroup.length > 0) groups.push(limitGroup)

  if (tokens) {
    const mark = tokens.isApprox ? '~' : ''
    const input = tokens.input + tokens.cacheWrite
    const allIn = input + tokens.cacheRead
    const contextLine =
      context && context.tokens !== null
        ? `Bağlam: ${formatTokens(context.tokens)} / ${formatTokens(context.window)} (${formatPercent(context.percent ?? (context.tokens / context.window) * 100)})`
        : context
          ? `Bağlam: — / ${formatTokens(context.window)}`
          : null
    const requestLine = `İstek: ${tokens.requests}`
    const approxLine = tokens.isApprox ? 'Döküm okunamadı: tur sonlarından yaklaşık toplam' : null
    const tail = [contextLine, requestLine, approxLine].filter((l): l is string => l !== null)

    groups.push([
      {
        kind: 'value',
        key: 'in',
        tone: 'red',
        icon: 'up',
        text: mark + formatTokens(input),
        title: [
          `Giriş: ${formatTokens(input)}`,
          `  önbelleksiz: ${formatTokens(tokens.input)}`,
          `  önbelleğe yazılan: ${formatTokens(tokens.cacheWrite)}`,
          ...tail,
        ].join('\n'),
      },
      {
        kind: 'value',
        key: 'out',
        tone: 'green',
        icon: 'down',
        text: mark + formatTokens(tokens.output),
        title: [
          `Çıkış: ${formatTokens(tokens.output)}`,
          tokens.requests > 0
            ? `  istek başına: ${formatTokens(tokens.output / tokens.requests)}`
            : null,
          ...tail,
        ]
          .filter((l): l is string => l !== null)
          .join('\n'),
      },
      {
        kind: 'value',
        key: 'cache',
        tone: 'blue',
        icon: 'layers',
        text: mark + formatTokens(tokens.cacheRead),
        title: [
          `Önbellekten okunan: ${formatTokens(tokens.cacheRead)}`,
          allIn > 0 ? `  tüm girişin ${formatPercent((tokens.cacheRead / allIn) * 100)}'i` : null,
          ...tail,
        ]
          .filter((l): l is string => l !== null)
          .join('\n'),
      },
    ])
  }

  if (costUsd !== null) {
    groups.push([
      {
        kind: 'value',
        key: 'cost',
        tone: 'gold',
        icon: 'dollar',
        text: `$${costUsd.toFixed(2)}`,
        title: [
          `Oturum maliyeti: $${costUsd.toFixed(2)}`,
          'API liste fiyatıyla hesaplanır; abonelikte faturaya yansımaz',
          tokens ? `İstek: ${tokens.requests}` : null,
        ]
          .filter((l): l is string => l !== null)
          .join('\n'),
      },
    ])
  }

  return groups
}

export function summaryLine(view: View): string {
  const parts: string[] = []
  for (const pill of pillGroups(view).flat()) {
    if (pill.kind === 'limit') {
      parts.push(`${pill.label} ${formatPercent(pill.percent)} (${pill.left})`)
    } else {
      const name = { in: 'giriş', out: 'çıkış', cache: 'önbellek', cost: 'maliyet' }[pill.key] ?? pill.key
      parts.push(`${name} ${pill.text}`)
    }
  }
  return parts.length > 0 ? parts.join(' · ') : 'Henüz kullanım verisi yok.'
}

// ---- SVG ----

const PALETTE: Record<Tone, { light: [string, string]; dark: [string, string] }> = {
  teal: { light: ['#d3f3ec', '#0b4d43'], dark: ['#14352f', '#9fe6d5'] },
  purple: { light: ['#ebe3fb', '#45297e'], dark: ['#2a2142', '#d3c1fa'] },
  red: { light: ['#fde1df', '#86201a'], dark: ['#3d1e1c', '#f5b3ae'] },
  green: { light: ['#dbf4dc', '#1b5c21'], dark: ['#1b3520', '#a6e5ab'] },
  blue: { light: ['#dce8fb', '#1c4888'], dark: ['#1a2b44', '#a8c7f4'] },
  gold: { light: ['#faefcc', '#6b4c00'], dark: ['#3a2f12', '#f1d179'] },
}

const BAR = { ok: '#1f9d68', warn: '#d9a400', high: '#e0464b' }

const FONT_SIZE = 11
const CHAR = FONT_SIZE * 0.6
const HEIGHT = 22
const PAD = 8
const ICON = 12

function esc(text: string): string {
  return text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
}

function icon(name: IconName | 'timer', x: number): string {
  const y = (HEIGHT - ICON) / 2
  const g = (body: string) =>
    `<g class="ic" transform="translate(${x} ${y})" fill="none" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">${body}</g>`
  switch (name) {
    case 'gauge':
      return g('<path d="M1.5 9.5a4.5 4.5 0 0 1 9 0"/><path d="M6 9.5 8.4 5.6"/><circle cx="6" cy="9.5" r=".9" class="dot"/>')
    case 'calendar':
      return g('<rect x="1.5" y="2.5" width="9" height="8" rx="1.5"/><path d="M1.5 5.5h9M4 1v3M8 1v3"/>')
    case 'timer':
      return g('<path d="M3 1.5h6M3 10.5h6M3.5 1.5c0 3 5 3 5 4.5S3.5 7.5 3.5 10.5M8.5 1.5c0 3-5 3-5 4.5s5 1.5 5 4.5"/>')
    case 'up':
      return g('<path d="M6 10.5v-9M2.5 5 6 1.5 9.5 5"/>')
    case 'down':
      return g('<path d="M6 1.5v9M2.5 7 6 10.5 9.5 7"/>')
    case 'layers':
      return g('<path d="M6 1.5 10.5 4 6 6.5 1.5 4z"/><path d="M1.5 6.5 6 9 10.5 6.5"/><path d="M1.5 8.8 6 11.3 10.5 8.8" opacity=".6"/>')
    case 'dollar':
      return g('<path d="M6 .8v10.4M8.6 3.2C8.1 2.4 7.2 2 6 2 4.5 2 3.4 2.8 3.4 3.9c0 2.6 5.4 1.4 5.4 4.2 0 1.1-1.1 1.9-2.8 1.9-1.3 0-2.3-.5-2.8-1.4"/>')
  }
}

function style(tone: Tone): string {
  const { light, dark } = PALETTE[tone]
  return `<style>
text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;font-size:${FONT_SIZE}px;fill:${light[1]}}
.bg{fill:${light[0]}}.ic{stroke:${light[1]}}.dot{fill:${light[1]};stroke:none}
.sep{stroke:${light[1]};opacity:.3}.track{fill:${light[1]};opacity:.16}.mark{fill:${light[1]}}
.b{font-weight:700}
@media (prefers-color-scheme: dark){
text{fill:${dark[1]}}.bg{fill:${dark[0]}}.ic{stroke:${dark[1]}}.dot{fill:${dark[1]}}
.sep{stroke:${dark[1]}}.track{fill:${dark[1]};opacity:.22}.mark{fill:${dark[1]}}
}
</style>`
}

export type PillSvg = { key: string; source: string; alt: string; width: number; height: number }

export function pillSvg(pill: Pill): PillSvg {
  const parts: string[] = []
  let x = PAD
  parts.push(icon(pill.icon, x))
  x += ICON + 5
  const baseline = 15

  if (pill.kind === 'limit') {
    parts.push(`<text x="${x}" y="${baseline}">${esc(pill.label)}</text>`)
    x += pill.label.length * CHAR + 6
    const barW = 38
    const barH = 5
    const barY = (HEIGHT - barH) / 2
    const fill = Math.min(1, Math.max(0, pill.percent / 100)) * barW
    parts.push(`<rect class="track" x="${x}" y="${barY}" width="${barW}" height="${barH}" rx="2.5"/>`)
    if (fill > 0) {
      parts.push(
        `<rect x="${x}" y="${barY}" width="${fill.toFixed(1)}" height="${barH}" rx="2.5" fill="${BAR[barLevel(pill.percent)]}"/>`,
      )
    }
    if (pill.elapsed !== null) {
      const mx = x + pill.elapsed * barW
      parts.push(`<rect class="mark" x="${(mx - 0.75).toFixed(1)}" y="${barY - 3}" width="1.5" height="${barH + 6}" rx=".75"/>`)
    }
    x += barW + 6
    const pct = formatPercent(pill.percent)
    parts.push(`<text class="b" x="${x}" y="${baseline}">${pct}</text>`)
    x += pct.length * CHAR + 7
    parts.push(`<line class="sep" x1="${x}" y1="5" x2="${x}" y2="${HEIGHT - 5}"/>`)
    x += 7
    parts.push(icon('timer', x - 1))
    x += ICON + 3
    parts.push(`<text x="${x}" y="${baseline}">${esc(pill.left)}</text>`)
    x += pill.left.length * CHAR
  } else {
    parts.push(`<text class="b" x="${x}" y="${baseline}">${esc(pill.text)}</text>`)
    x += pill.text.length * CHAR
  }

  const width = Math.ceil(x + PAD + 1)
  const source =
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${HEIGHT}" viewBox="0 0 ${width} ${HEIGHT}">` +
    style(pill.tone) +
    `<g><title>${esc(pill.title)}</title>` +
    `<rect class="bg" x="0" y="0" width="${width}" height="${HEIGHT}" rx="${HEIGHT / 2}"/>` +
    parts.join('') +
    `</g></svg>`
  return { key: pill.key, source, alt: pill.title.replaceAll('\n', '; '), width, height: HEIGHT }
}

// ---- Terminal ----

export const TERMINAL_COLOR: Record<Tone, string> = {
  teal: '#2bb5a0',
  purple: '#a07ff0',
  red: '#ec6a63',
  green: '#4cc35a',
  blue: '#5a9ef0',
  gold: '#e3b33c',
}

export const TERMINAL_BAR = BAR

export const TERMINAL_ICON: Record<IconName, string> = {
  gauge: '◔',
  calendar: '▦',
  up: '↑',
  down: '↓',
  layers: '≋',
  dollar: '$',
}

/** 10 hücrelik çubuk: dolu kısım, boş kısım ve zaman çizgisinin yeri. */
export function terminalBar(pill: LimitPill, cells = 10): { filled: string; empty: string; markAt: number | null } {
  const filledCount = Math.round(Math.min(1, Math.max(0, pill.percent / 100)) * cells)
  const markAt = pill.elapsed === null ? null : Math.min(cells - 1, Math.floor(pill.elapsed * cells))
  return { filled: '█'.repeat(filledCount), empty: '░'.repeat(cells - filledCount), markAt }
}
