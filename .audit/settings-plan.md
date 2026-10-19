# Settings implementation

## Workflow

- [x] Read poteto-mode Principles.
- [x] Frame. Confirm current engine, desktop and build boundaries.
- [x] Ground. Read-only engine and UI exploration.
- [x] Sketch. Compare shared typed JSON settings and UI-owned CLI mappings.
- [x] Agree. Choose shared standard-library-only settings package and immutable JSON command argument.
- [x] Implement. Settings contract, engine behavior, classified desktop page.
- [x] Scrap. Not needed; shared settings boundary survived numeric-parameter changes.
- [x] Verify. Build artifacts and repeatable archive/codec/settings checks passed. Final native UI inspection intentionally skipped after the user's focus restriction.
- [x] Review. Independent WIP review found hard-link dependency and an unjustified C-drive restriction. Both removed. Final build and static checks passed.

## Throughput checkpoint

- Blocking first steps. Fix the configuration contract and document failure cases before implementation.
- Independent workstreams. Codec support can run separately from UI and engine integration.
- Shared mutable state. Assign disjoint files; integrate configuration contract before callers.
- Smallest safe decomposition. One engine owner, one codec owner, main agent owns UI and integration.

## UI checklist

- [x] Brief and mode established. Refine the existing MyGo desktop app.
- [x] Visual direction chosen. Preserve existing light palette and typography; category navigation and scrollable settings body.
- [x] Interface built with real content, locked running state, validation and persistence errors.
- [x] Initial native UI rendered and inspected. Later sidebar/footer/numeric changes are source/build checked only because the user prohibited focus stealing.
- [x] Result built at bin/mangatrans-ui-settings.exe with updated sibling engine. Existing UI was not restarted.

## Failure cases to exercise

- Missing settings file uses defaults; malformed or invalid settings are reported without overwriting the file.
- Unsupported enum, out-of-range quality, invalid model/font path fail before inference.
- Settings survive restart; a task uses an immutable configuration snapshot.
- Excluded controls never appear on the home or settings pages.
- JPEG, PNG, lossy WebP and lossless WebP output decode and have matching filenames.
- WebP input inside CBZ is decoded; archive order, page count and unique names survive conversion.
- Blank and failed pages retain original bytes when required.
- Cache invalidation follows input/model/settings changes; style-only changes can reuse recognition and translation.
- No failed page gets text painted over an unclean original image.
- Existing output requires confirmation or a collision-free automatic name.
- Cancellation does not destroy the user's source or an existing successful output.

## User corrections

- Remove the language/local-processing text from the sidebar.
- Align compact navigation at the top and left; remove white card buttons.
- Remove the entire footer, including its instructions, restore button and duplicate home button.
- Show numerical parameter values and units rather than Chinese quality tiers. Mark default values.
- Do not launch, activate or manipulate native UI windows during further verification.
- Investigate the antivirus alert without disabling protection or restoring quarantined files.
