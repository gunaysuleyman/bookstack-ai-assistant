# usage-band kurulumu (Windows PowerShell 5.1 ve PowerShell 7).
# Tek satırla çalışır:
#   irm https://raw.githubusercontent.com/gunaysuleyman/bookstack-ai-assistant/claude/inspiring-hamilton-x1pufe/mods/usage-band/install.ps1 | iex
# Yaptıkları:
#   1. Mod dosyalarını ~/.claude/mods/usage-band altına indirir.
#   2. ~/.claude/settings.json'daki env bloğuna CLAUDE_CODE_PLUGIN_DIRS ve
#      CLAUDE_CODE_ENABLE_FUNCTION_HOOKS ekler; diğer ayarlara dokunmaz, önce yedek alır.
#   3. Node.js yoksa winget ile kurmayı dener (token sayımı için gerekli).
#   4. claude komutu varsa "claude -p /kullanim" ile doğrular.

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$Base = 'https://raw.githubusercontent.com/gunaysuleyman/bookstack-ai-assistant/claude/inspiring-hamilton-x1pufe/mods/usage-band'
$Files = @(
  '.claude-plugin/plugin.json',
  'hooks/hooks.json',
  'hooks/register.tsx',
  'hooks/pills.ts',
  'scripts/tokens.mjs',
  'types/index.d.ts',
  'README.md'
)

$Home_ = if ($env:USERPROFILE) { $env:USERPROFILE } else { $HOME }
$ClaudeDir = Join-Path $Home_ '.claude'
$ModDir = Join-Path (Join-Path $ClaudeDir 'mods') 'usage-band'
$Settings = Join-Path $ClaudeDir 'settings.json'
$Sep = [IO.Path]::PathSeparator
$Utf8 = New-Object System.Text.UTF8Encoding($false)

function Say($text) { Write-Host "[usage-band] $text" }

# 1. Dosyalar
Say "Dosyalar indiriliyor: $ModDir"
foreach ($rel in $Files) {
  $target = Join-Path $ModDir ($rel -replace '/', [IO.Path]::DirectorySeparatorChar)
  $dir = Split-Path $target -Parent
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
  Invoke-WebRequest -UseBasicParsing -Uri "$Base/$rel" -OutFile $target
}

# 2. settings.json
if (-not (Test-Path $ClaudeDir)) { New-Item -ItemType Directory -Force -Path $ClaudeDir | Out-Null }
if (Test-Path $Settings) {
  $raw = [IO.File]::ReadAllText($Settings, $Utf8)
  $backup = "$Settings.bak-usage-band"
  if (-not (Test-Path $backup)) {
    [IO.File]::WriteAllText($backup, $raw, $Utf8)
    Say "settings.json yedeklendi: $backup"
  }
  if ($raw.Trim().Length -eq 0) { $config = New-Object PSObject } else { $config = $raw | ConvertFrom-Json }
} else {
  $config = New-Object PSObject
}

if (-not ($config.PSObject.Properties.Name -contains 'env') -or $null -eq $config.env) {
  $config | Add-Member -Force -NotePropertyName 'env' -NotePropertyValue (New-Object PSObject)
}
$envBlock = $config.env

$current = $null
if ($envBlock.PSObject.Properties.Name -contains 'CLAUDE_CODE_PLUGIN_DIRS') { $current = [string]$envBlock.CLAUDE_CODE_PLUGIN_DIRS }
if ([string]::IsNullOrWhiteSpace($current)) {
  $dirs = $ModDir
} elseif (($current.Split($Sep) | ForEach-Object { $_.Trim() }) -contains $ModDir) {
  $dirs = $current
} else {
  $dirs = $current.TrimEnd($Sep) + $Sep + $ModDir
}
$envBlock | Add-Member -Force -NotePropertyName 'CLAUDE_CODE_PLUGIN_DIRS' -NotePropertyValue $dirs
$envBlock | Add-Member -Force -NotePropertyName 'CLAUDE_CODE_ENABLE_FUNCTION_HOOKS' -NotePropertyValue '1'

[IO.File]::WriteAllText($Settings, ($config | ConvertTo-Json -Depth 64), $Utf8)
Say "settings.json güncellendi: $Settings"

# 3. Node.js
$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node -and (Test-Path 'C:\Program Files\nodejs\node.exe')) { $node = 'C:\Program Files\nodejs\node.exe' }
if (-not $node) {
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    Say 'Node.js bulunamadı, winget ile kuruluyor...'
    winget install --id OpenJS.NodeJS.LTS -e --silent --accept-package-agreements --accept-source-agreements | Out-Null
  } else {
    Say 'UYARI: Node.js bulunamadı. Token sayıları "~" ile yaklaşık gösterilecek. https://nodejs.org adresinden kurabilirsiniz.'
  }
} else {
  Say 'Node.js bulundu.'
}

# 4. Doğrulama
$claude = Get-Command claude -ErrorAction SilentlyContinue
if ($claude) {
  Say 'Doğrulanıyor: claude -p "/kullanim"'
  $env:CLAUDE_CODE_PLUGIN_DIRS = $dirs
  $env:CLAUDE_CODE_ENABLE_FUNCTION_HOOKS = '1'
  & claude -p '/kullanim'
} else {
  Say 'claude komutu PATH''te yok; doğrulama atlandı (masaüstü uygulaması için gerekmez).'
}

Say 'Tamam. Claude masaüstü uygulamasını tamamen kapatıp yeniden açın; şerit ilk yanıttan sonra mesaj kutusunun üstünde görünür.'
