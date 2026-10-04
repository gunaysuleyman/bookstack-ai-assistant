# usage-band

Claude Code mod'u: mesaj kutusunun üstünde kullanım şeridi.
`[5h 7d] [giriş çıkış önbellek] [maliyet]` hapları; masaüstünde SVG, terminalde renkli metin.

- `/kullanim`: verileri yenile, tek satır özet
- `/kullanim gizle` / `/kullanim goster`: şeridi kapat / aç

Token toplamları `scripts/tokens.mjs` ile oturum dökümünden sayılır
(`node` PATH'te, `/usr/local/bin`, `/opt/homebrew/bin` ya da `C:\Program Files\nodejs` altında olmalı);
çalışmazsa tur sonlarındaki değerler `~` ile gösterilir.

## Kalıcı kurulum (Windows)

[Node.js](https://nodejs.org) kurulu olmalı. PowerShell'de, deponun kökünden:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.claude\mods" | Out-Null
Copy-Item -Recurse -Force mods\usage-band "$env:USERPROFILE\.claude\mods\"
```

`%USERPROFILE%\.claude\settings.json` içindeki `env` bloğuna (`<kullanıcı>` yerine kendi adınız):

```json
"CLAUDE_CODE_PLUGIN_DIRS": "C:\\Users\\<kullanıcı>\\.claude\\mods\\usage-band",
"CLAUDE_CODE_ENABLE_FUNCTION_HOOKS": "1"
```

(`CLAUDE_CODE_PLUGIN_DIRS` zaten varsa yolu `;` ile sona ekleyin.) Sonra uygulamayı yeniden başlatın ve
`claude -p "/kullanim"` ile doğrulayın.

## Kalıcı kurulum (macOS/Linux)

```sh
mkdir -p ~/.claude/mods && cp -R mods/usage-band ~/.claude/mods/
```

`~/.claude/settings.json` içindeki `env` bloğuna:

```json
"CLAUDE_CODE_PLUGIN_DIRS": "/Users/<kullanıcı>/.claude/mods/usage-band",
"CLAUDE_CODE_ENABLE_FUNCTION_HOOKS": "1"
```

(`CLAUDE_CODE_PLUGIN_DIRS` zaten varsa yolu `:` ile sona ekleyin.) Doğrulama:

```sh
claude plugin validate ~/.claude/mods/usage-band
claude plugin test ~/.claude/mods/usage-band
claude -p "/kullanim"
```
