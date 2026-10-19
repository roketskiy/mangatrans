# Project verification

- Do not launch, activate, focus, click, or send keys to the native desktop app during automated verification. The user works on this desktop. Use background builds and checks. Ask before any interactive check.
- The host antivirus blocked a Go-generated `translate.test.exe` during this task. Do not run `go test` on this host until that alert is resolved. Do not disable antivirus, add exclusions, restore quarantined files, or rename a flagged executable to run it.
- Use the named verification commands in `cmd/verify-settings` and `cmd/verify-pageimage`, static checks, and bounded engine runs. A build or codec check does not prove a full translation run.
- Settings show numeric parameters and units directly. Show a default label next to the default value. Do not replace numeric parameters with vague quality presets.
- Home exposes only the settings entry point. Do not add performance presets, OCR concurrency, glossary, or page-limit controls to the settings UI.
