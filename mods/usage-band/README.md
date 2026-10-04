# usage-band

Claude Code mod'u: mesaj kutusunun üstünde kullanım şeridi.
`[5h 7d] [giriş çıkış önbellek] [maliyet]` hapları; masaüstünde SVG, terminalde renkli metin.

- `/kullanim`: verileri yenile, tek satır özet
- `/kullanim gizle` / `/kullanim goster`: şeridi kapat / aç

Token toplamları `scripts/tokens.mjs` ile oturum dökümünden sayılır
(`node` PATH'te, `/usr/local/bin` ya da `/opt/homebrew/bin` altında olmalı);
çalışmazsa tur sonlarındaki değerler `~` ile gösterilir.

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
