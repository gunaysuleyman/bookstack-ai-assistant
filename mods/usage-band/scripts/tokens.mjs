#!/usr/bin/env node
// Bir oturumun token toplamlarını dökümünden sayar.
// Kullanım: node tokens.mjs <oturum-kimliği>
// Okur: ~/.claude/projects/*/<kimlik>.jsonl ve ~/.claude/projects/*/<kimlik>/subagents/*.jsonl
// Yazar (stdout, tek satır JSON):
//   { path, size, mtimeMs, input, cacheWrite, output, cacheRead, requests }
// Aynı mesaj her içerik bloğu için tekrar yazıldığından message.id + requestId
// anahtarıyla tekilleştirilir ve her alanın en büyüğü alınır.
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { join } from 'node:path'

const id = process.argv[2]
if (!id || !/^[\w-]+$/.test(id)) {
  console.log(JSON.stringify({ error: 'oturum kimliği eksik ya da geçersiz' }))
  process.exit(2)
}

const projects = join(process.env.CLAUDE_CONFIG_DIR || join(homedir(), '.claude'), 'projects')

function findFiles() {
  let main = null
  const files = []
  for (const dir of existsSync(projects) ? readdirSync(projects) : []) {
    const file = join(projects, dir, `${id}.jsonl`)
    if (existsSync(file)) {
      main ??= file
      files.push(file)
    }
    const subagents = join(projects, dir, id, 'subagents')
    if (existsSync(subagents)) {
      for (const name of readdirSync(subagents)) {
        if (name.endsWith('.jsonl')) files.push(join(subagents, name))
      }
    }
  }
  return { main, files }
}

const { main, files } = findFiles()
if (main === null) {
  console.log(JSON.stringify({ error: `döküm bulunamadı: ${id}` }))
  process.exit(3)
}

const stats = files.map(file => {
  const { size, mtimeMs } = statSync(file)
  return { file, size, mtimeMs }
})
const signature = JSON.stringify(stats)
const cacheFile = join(tmpdir(), `usage-band-${id}.json`)

try {
  const cached = JSON.parse(readFileSync(cacheFile, 'utf8'))
  if (cached.signature === signature) {
    console.log(JSON.stringify(cached.result))
    process.exit(0)
  }
} catch {}

const FIELDS = ['input_tokens', 'cache_creation_input_tokens', 'output_tokens', 'cache_read_input_tokens']
const byKey = new Map()

for (const { file } of stats) {
  for (const line of readFileSync(file, 'utf8').split('\n')) {
    if (!line.includes('"usage"')) continue
    let row
    try {
      row = JSON.parse(line)
    } catch {
      continue
    }
    const usage = row?.message?.usage
    if (row?.type !== 'assistant' || !usage) continue
    const key = `${row.message.id ?? row.uuid}:${row.requestId ?? ''}`
    const seen = byKey.get(key) ?? Object.fromEntries(FIELDS.map(f => [f, 0]))
    for (const f of FIELDS) seen[f] = Math.max(seen[f], Number(usage[f]) || 0)
    byKey.set(key, seen)
  }
}

const sum = f => [...byKey.values()].reduce((total, u) => total + u[f], 0)
const result = {
  path: main,
  size: statSync(main).size,
  mtimeMs: statSync(main).mtimeMs,
  input: sum('input_tokens'),
  cacheWrite: sum('cache_creation_input_tokens'),
  output: sum('output_tokens'),
  cacheRead: sum('cache_read_input_tokens'),
  requests: byKey.size,
  files: stats.length,
}

try {
  writeFileSync(cacheFile, JSON.stringify({ signature, result }))
} catch {}

console.log(JSON.stringify(result))
