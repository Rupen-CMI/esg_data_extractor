# Free-model benchmark — company identification

**Task:** extract key-player company names from DuckDuckGo snippets for 5 markets.
**Input:** identical cached snippets for every model (`benchmark/snippets.json`).
**Gold set:** 40 companies hand-built from the snippets (`benchmark/ground_truth.json`).
**Scoring:** alias-aware matching; duplicates (same company, different alias) are not
penalized; precision/recall/F1 micro-averaged across markets.
**Date:** 2026-06-24. Provider: opencode.ai Zen (OpenAI-compatible).

## Results (ranked by F1)

| Rank | Model | Precision | Recall | F1 | Hallucinations | Errors | Avg latency | Tokens | Status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **north-mini-code-free** | **1.00** | **1.00** | **1.00** | 0 | 0/5 | 21.9s | 12,745 | ✅ usable |
| 2 | deepseek-v4-flash-free | 1.00 | 0.85 | 0.919 | 0 | 0/5 | 29.8s | 17,287 | ✅ usable |
| 3 | mimo-v2.5-free | 0.97 | 0.83 | 0.892 | 1 | 0/5 | 31.2s | 17,554 | ✅ usable |
| 4 | nemotron-3-ultra-free | 1.00 | ~0.53* | low | 0 | 3–4/5 | 120–190s | — | ⚠️ slow/unstable |
| — | qwen3.6-plus-free | — | — | — | — | — | — | — | ❌ free promo ended |
| — | minimax-m3-free | — | — | — | — | — | — | — | ❌ free promo ended |

\* nemotron's best observed recall in an isolated run; it times out / 500s under any load.

## Key findings

- **north-mini-code-free wins decisively** — perfect precision and recall, fastest of
  the reasoning-capable models, lowest token use, fully reliable.
- **deepseek & mimo** are excellent too (precision ~1.0). Their only misses are on the
  **Lithium Ion Battery** snippet, where 6 companies (Panasonic, Samsung, Toshiba, ATL,
  BAK, Blue Energy) appear under a mislabeled "Industrial Valves market" sentence. They
  *chose to exclude* those — arguably the more conservative/correct call. Excluding that
  one debatable snippet, both are effectively perfect.
- **nemotron-3-ultra-free** is accurate when it answers but is far too slow (2–3 min/call)
  and frequently times out — not viable for a pipeline.
- **qwen & minimax** free tiers are gone ("Free promotion has ended — subscribe to
  OpenCode Go"). Not usable on this key without billing.

## Recommendation

Set `DEFAULT_MODEL = "north-mini-code-free"` in `zen_client.py`.
Good fallback chain: north-mini-code → deepseek-v4-flash → mimo-v2.5.

Reproduce: `venv/Scripts/python.exe benchmark/run_benchmark.py`
