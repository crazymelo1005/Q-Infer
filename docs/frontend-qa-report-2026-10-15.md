# Frontend QA Report — 2026-10-15

**Scope:** Full browser audit of Q-Infer web UI (Chat / Monitor / Requests / About tabs), design-token compliance, accessibility, theme, i18n, and CSS design debt.

**Method:** Live server with fixture model (`qinfer-serve-fixture.gguf`), browser-use MCP DOM inspection, computed-style analysis, WCAG contrast calculation, CSS source audit.

**Verdict: Pass with warnings** — no blockers; several major polish issues and significant design-token debt.

---

## Major Issues

### M1. Favicon color mismatch

The favicon SVG in `index.html:7` uses `fill='%2310b981'` (emerald green). The design system accent is `#f2600c` (orange). This is the first thing users see in their tab bar — it reads as a different product.

- **File:** `src/serve/web/index.html` line 7
- **Fix:** Change `%2310b981` to `%23f2600c`

### M2. 8 form inputs lack accessible labels

Screen readers will announce these inputs only by placeholder (which disappears on focus) or not at all:

| Input | Issue |
|---|---|
| `#conv-search` | placeholder only, no `<label>` or `aria-label` |
| `#conv-import-file` | file input, no label at all |
| `#sys-prompt` | placeholder only |
| `#chat-input` | placeholder only |
| `#req-search` | placeholder only |
| `#s-override` | checkbox inside `<label>` text — OK structurally but the label text is in a `<span>` with `data-i18n`, verify screen reader reads it |
| 2 unnamed checkboxes | no id, no aria-label (inside sampler drawer and tool panel) |

- **Fix:** Add `aria-label` attributes (simplest) or associate `<label>` elements.

### M3. Conversation sidebar action buttons lack `aria-label`

Each conversation item has 3 icon-only buttons (pin / rename / delete) with `title` attributes but no `aria-label`. Screen readers may not announce `title` on buttons consistently.

- **File:** `src/serve/web/app.js` (where conv items are rendered)
- **Fix:** Add `aria-label` mirroring the `title` text.

### M4. Missing CSS token definitions (referenced but undefined)

These tokens are used as fallbacks in CSS but never defined in `tokens.css`:

| Token | Where referenced | Effect |
|---|---|---|
| `--st-fs-2xs` | `app.css:482,490` | `.st-kbd`, `.conv-group-label` fall back to hardcoded `11px` |
| `--st-fs-3xs` | not yet referenced | needed for `10px` elements (badges, spark axis) |
| `--st-scrim` | `app.css:472` | scrim overlay falls back to hardcoded `rgba(15,23,42,.45)` |
| `--st-shadow-lg` | `app.css:473` | shadow falls back to hardcoded value |

- **Fix:** Define these tokens in `tokens.css` `:root` and `[data-theme="dark"]`.

---

## Minor Issues

### m1. Slider output format mismatch

`presence_penalty` and `frequency_penalty` have `step="0.1"` (1 decimal place) but their `<output>` elements display `"0.00"` (2 decimal places). This suggests the values can only change in 0.1 increments but appear to have 0.01 precision — misleading.

- **Fix:** Either change `step` to `0.01` or format output to 1 decimal place.

### m2. Context length options don't cover engine maximum

The `<select id="s-ctx">` offers up to 64K, but the engine supports 262144 (256K). Users can't select the full range.

- **Fix:** Add `128K` and `256K` options, or switch to a numeric input.

### m3. Bubble role line `margin-bottom: 2px` is too tight

The gap between the role/timestamp line and the bubble body is 2px — much tighter than the 8px gap within the role line itself. This creates a cramped feel.

- **File:** `app.css:28`
- **Fix:** Use `var(--st-s-1)` (4px) or `var(--st-s-2)` (8px).

### m4. Tiny fonts below 12px

10 elements use `font-size: 10px` (spark axis labels, float badge, tool count). These are below the WCAG recommended minimum of 12px for body text.

- **Impact:** Low — these are decorative/badge elements, not readable text. But users with low vision may struggle.

### m5. Dark theme accent-tint background is non-opaque

The dark theme `--st-accent-tint` is `rgba(255,138,61,.14)` — a semi-transparent color. The foreground `--st-accent-text` is `#ffb887`. The actual contrast depends on what's behind the tint (which varies). This could fail WCAG on some backgrounds.

- **Fix:** Consider using an opaque tint for dark theme, or verify contrast against both `--st-bg` and `--st-surface`.

---

## Design Debt

### d1. Spacing scale violations (~50+ instances)

The design system defines a 4pt grid (`--st-s-1:4px` through `--st-s-16:64px`), but the CSS frequently uses values that don't fit:

| Value | Occurrences | Suggested token |
|---|---|---|
| 6px | ~15 | `--st-s-1h: 6px` (half-step) |
| 10px | ~10 | `--st-s-2h: 10px` |
| 14px | ~7 | `--st-s-3h: 14px` |
| 18px | 3 | between `--st-s-4`(16) and `--st-s-5`(20) |
| 22px | 1 | between `--st-s-5`(20) and `--st-s-6`(24) |
| 30px | 2 | between `--st-s-6`(24) and `--st-s-8`(32) |
| 42px, 44px | 1 each | between `--st-s-10`(40) and `--st-s-12`(48) |

### d2. Border-radius violations (~12 instances)

| Value | Used for | Suggested token |
|---|---|---|
| 6px | bubble tail corners (6 places) | `--st-r-corner: 6px` |
| 9px | badges, queue count | `--st-r-badge: 9px` |
| 10px | tab buttons, icon buttons | `--st-r-sm` (8px) or new token |

### d3. Font-size violations (~12 instances)

| Value | Used for | Suggested token |
|---|---|---|
| 11px | role text, badges, conv time | `--st-fs-2xs: 11px` |
| 10px | spark axis, float badge, tool count | `--st-fs-3xs: 10px` |

### d4. Hardcoded colors in components

| File | Value | Context |
|---|---|---|
| `components.css:84` | `rgba(255,255,255,.18)` | button inset highlight |
| `components.css:101` | `#fff` | toggle knob |
| `components.css:117` | `rgba(15,23,42,.18)` | scrim background |
| `app.css:71-72` | `rgba(255,255,255,.18/.28)` | user bubble action hover/active |

### d5. Legacy token name fallback

`tokens.css:191,216` reference `--st-border` as a fallback variable name instead of `--st-line`.

---

## Accessibility — Passing

### WCAG Contrast Ratios (Light Theme)

| Pair | Ratio | AA | AA Large |
|---|---|---|---|
| Body ink / bg | 15.78 | pass | pass |
| Body ink-soft / bg | 12.63 | pass | pass |
| Muted ink / bg | 4.81 | pass | pass |
| Accent ink / accent-strong | 4.70 | pass | pass |
| Body ink / surface | 16.49 | pass | pass |
| Muted ink / surface | 5.02 | pass | pass |
| Code ink / code-bg | 13.60 | pass | pass |
| Accent text / accent-tint | 4.71 | pass | pass |
| Danger text / danger-tint | 6.07 | pass | pass |
| Info text / info-tint | 6.11 | pass | pass |
| Warn text / warn-tint | 5.51 | pass | pass |

### WCAG Contrast Ratios (Dark Theme)

| Pair | Ratio | AA | AA Large |
|---|---|---|---|
| Body ink / bg | 16.70 | pass | pass |
| Muted ink / bg | 6.39 | pass | pass |
| Muted ink / surface | 5.90 | pass | pass |
| Accent ink / accent-strong | 7.53 | pass | pass |
| Code ink / code-bg | 16.08 | pass | pass |
| Body ink / surface | 15.43 | pass | pass |

### Focus States

93 focusable elements; only 1 without explicit focus styling (acceptable — it has `outline: none` via reset but the browser default outline applies).

### Keyboard Navigation

- Tab order follows visual layout
- All interactive elements reachable
- Escape closes modals/popovers
- Keyboard shortcuts modal (Ctrl+?) works correctly with 7 shortcuts documented

---

## Verified Working

| Feature | Status | Notes |
|---|---|---|
| Tab navigation (Chat/Monitor/Requests/About) | pass | All 4 tabs switch correctly |
| Theme toggle (light/dark) | pass | Colors update, localStorage persists |
| i18n toggle (EN/ZH) | pass | All `data-i18n` elements update |
| Chat message send/receive | pass | Test message sent, response streamed |
| Chat status bar | pass | Shows state, queue count, decode speed |
| Status pill | pass | State changes (idle/generating), dot color updates |
| Sampler drawer | pass | All sliders, presets, override switch functional |
| Thinking level control | pass | 4 levels (off/low/medium/xhigh), segmented control |
| System prompt drawer | pass | Opens/closes, flag indicator works |
| Conversation sidebar | pass | 4 conversations listed, search, new/import/export buttons |
| Conversation actions | pass | Pin/rename/delete per conversation |
| User bubble actions | pass | Copy, edit, branch from here |
| Assistant bubble actions | pass | Copy, regenerate, thumbs up, thumbs down |
| Message queue panel | pass | Hidden when empty, shows when queued |
| Tool panel | pass | Toggle via button, shows tool list |
| Keyboard shortcuts modal | pass | Opens with ? or button, 7 shortcuts listed |
| Monitor tab | pass | State card, 11 metrics, 5 state badges, context gauge |
| Requests tab | pass | Session counters, conversation cache, request table, inspector |
| About tab | pass | 5 cards (connection, model, capabilities, hardware, service) |
| API key input | pass | Password type, show/hide toggle |
| Markdown rendering | pass | Code blocks, tables, task lists, blockquotes |
| Code syntax highlighting | pass | Keywords, strings, numbers, comments, functions |
| No inline style color debt | pass | 0 hardcoded inline colors in DOM |

---

## Recommendations (Priority Order)

1. **Fix favicon** — 1 line change, immediate brand consistency
2. **Add aria-labels** to all icon-only buttons and inputs without labels
3. **Define missing tokens** (`--st-fs-2xs`, `--st-fs-3xs`, `--st-scrim`, `--st-shadow-lg`)
4. **Fix slider format** — align `step` and display precision for penalty sliders
5. **Add 128K/256K context options** to the select dropdown
6. **Incremental design debt cleanup** — add half-step spacing tokens (6px, 10px, 14px) and corner radius token (6px), then replace hardcoded values

---

*Report generated 2026-10-15 from live browser inspection of Q-Infer web UI at localhost:8080.*
