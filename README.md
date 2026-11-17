# Small local LLMs on an 8 GB laptop GPU: speed, honest evaluation, and training-free weight surgery

A measured log from one afternoon of experiments (17 Nov 2026) on an **RTX 4060 Laptop (8 GB, 256 GB/s)**, Windows 11,
`llama.cpp` b12473 (official CUDA 12.4 build). Everything is reproducible with the scripts in `scripts/`; model weights
are **not** included (the `scripts/indir_*.sh` downloaders fetch each file from Hugging Face / GitHub releases and verify its SHA-256). Numbers are single-machine, single-run unless a ± is given.
Negative results are included on purpose.

## TL;DR (all measured)

| What | Result |
|---|---|
| Plain single-stream decode, Llama 3.2 3B | Q8_0 **68.5** tok/s, Q4_K_M **106.5** tok/s (84-91 % of the memory-bandwidth ceiling) |
| Lossless speculative decoding (3B Q4_K_M + model-specific DSpark draft, block 6) | code/math **≈305**, chat **≈163**, story **≈122** tok/s (plain: ≈105) |
| Same + n-gram speculation on copy-heavy work (code editing) | **1,089 - 1,152** tok/s (plain: 99) |
| 16 concurrent requests, real `llama-server` (`-np 16 -kvu -bs`) | Q8_0 **728**, Q4_K_M **977** tok/s aggregate (default flags: 298 / 411) |
| Quality cost of Q4_K_M vs Q8_0 (wikitext-2, 4k tokens) | mean KL **0.0195 ± 0.0006**, same top-1 token 92.9 %, PPL ratio 1.021 |
| Speculative decoding vs plain decoding | output token-identical in 6 of 9 runs; other 3 diverge at a near-tie token (floating-point order) |
| Scale reference: Llama 3.2 **1B** + its DSpark draft | code 503, math 590, repeat 564 tok/s single stream |

**What this does *not* show:** a 3-4 B model replacing a frontier model, or "intelligence jumps" from weight edits.
See the weight-surgery section for what the data actually supports.

## 1. Speed

* `llama-bench` (FA on, 5 reps): Llama-3.2-3B-Instruct Q8_0 = 68.5 tok/s, Q4_K_M = 106.5 tok/s, pp512 ≈ 5.1-5.5 k tok/s.
* Raw batching (`llama-batched-bench`, prompt 128 + 128 generated, decode only), tok/s:

| streams | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Q8_0 | 68 | 128 | 245 | 458 | 818 | 1,314 | 1,848 |
| Q4_K_M | 105 | 199 | 364 | 672 | 1,129 | 1,553 | 1,927 |

* Real server, 128 tokens/request: the unified KV cache (`-kvu`) and GPU sampling (`-bs`) were worth ~2.4x at 16 requests
  (298 -> 728 for Q8_0, 411 -> 977 for Q4_K_M). A q8_0 KV cache was ~30 % slower than f16 here.
* Speculative decoding (`--spec-type`): a generic 1B draft helps only on predictable text and can *hurt* chat on a
  fast Q4 target. A draft trained for the exact target (DSpark, `rasyosef/Llama-3.2-3B-Instruct-DSpark-GGUF`, re-quantised
  to Q4_0 for +10 %) gives 1.5-2.9x. Backend sampling +4-5 %.
* Cycle-time ceiling (synthetic acceptance, output invalid, timing only): block 8 at mean acceptance 4.5 / 6.5 / 8.0
  gives 296 / 377 / 489 tok/s; i.e. on this card one verify+draft cycle is ≈15-16 ms, so ≥ 8 accepted tokens per cycle
  would be needed for 500 tok/s on *general* text. Copy-heavy text (code editing, quoting) does exceed 1,000 tok/s.
* n-gram speculation learns from earlier requests on the same server: replaying the *same* prompt shows fake 1,642 tok/s.
  All numbers in this repo are fresh-server, first-run.

## 2. Evaluation harness (`scripts/degerlendir.py`, `scripts/kod_degerlendir.py`)

Same scorer for every model: GSM8K (answer-is-N / last number), MATH-500 (`\boxed{}` + light normalisation), HumanEval
(sandboxed subprocess, AST deny-list). Reports accuracy, **truncation**, **repetition-loop** and **no-final-answer** rates.

Llama 3.2 3B Instruct Q8_0, greedy, zero-shot CoT:

| set | n | accuracy | truncated | looped | no final answer |
|---|---|---|---|---|---|
| GSM8K | 300 | 81.7 % (± 4.4) | 0.7 % | 1.7 % | 0.7 % |
| MATH-500 (first 200) | 200 | 40.5 % (± 6.8) | 15.0 % | 13.5 % | 24.0 % |
| MATH-500 (held-out 200-499) | 300 | 40.7 % (± 5.6) | 12.3 % | 14.0 % | 19.7 % |

If every truncated/looped answer were solved at the rate of the clean ones, MATH-500 would reach 47.2 % (+6.7 points):
that is the *ceiling* for any "remove the defect" intervention on this model.

## 3. Training-free weight surgery ("Weight-Space Programming")

**Setting.** A recent write-up reports that an LLM edited ~131 k weights of Qwen3.5-0.8B (0.016 %) and raised MATH-500
from 7.9 % to 48.1 %, with a control model (no loops) gaining ~1 point. The claimed mechanism: *the base model loops in
thinking mode; removing a "repetition direction" lets it finish*.

**What we measured.**

1. *Decoding the edit.* Diffing the public base GGUF against the edited GGUF (both Q8_0): exactly **two tensors differ**,
   `blk.15.ffn_down` and `blk.19.ffn_down`; each difference is **exactly rank-1** (rank-1 energy 1.000).
   It is `W' = W - 6 u (uᵀ W)` restricted to the ~64 neurons with the largest `|uᵀW|`, i.e. the components of those
   neurons' outputs along `u` are flipped and amplified ×5. `u` (unit, 1024-d) is similar in both layers (|cos| = 0.85).
   Qwen3.5 is a hybrid: 18 gated-DeltaNet + 6 full-attention blocks (layers 3,7,11,15,19,23).
2. *Base behaviour* (thinking mode, T = 0.6, top-p 0.95, top-k 20, 8,192-token cap, n = 100): **0 % correct,
   100 % hit the cap, 84 % looped.**
3. *Edited model*, same settings, first 40 problems: loops **80 % -> 32.5 %** (real effect), truncation 100 % -> 97.5 %,
   accuracy 0 % -> 2.5 % (1/40). At an 8k budget the edit removes most loops but the chains are still too long to finish;
   the write-up's own completed answers have a median of ~17k tokens, so the headline accuracy needs a far larger budget.
4. *Our own recipe* (no PyTorch): `llama-imatrix` on loop-region text vs early-reasoning text of the same answers ->
   mean difference of MLP inputs per layer -> loop direction -> rank-1 LoRA GGUF applied at run time
   (`scripts/wsp_kes.py`). Our direction vs. the edit's `u`: |cos| = 0.66 (L15) and 0.77 (L19) (the residual stream has a
   strong shared component: mean |cos| over all layers is 0.56). Applied with the same recipe (L15+19, K=64, ×6) it did
   **not** reduce loops (90 %, no better than base); a random direction of the same shape changes perplexity by +0.2 %,
   ours by +3.4 % (so the adapter is applied and does something, just not the right thing).

5. *Lottery scan of candidate edits* (Qwen3.5-0.8B, thinking mode, T = 0.6, first 16 MATH-500 problems, 8,192-token cap; every candidate is a rank-1 LoRA GGUF per layer, applied at run time). "Variance-weighted" = diagonal-LDA direction from the same imatrix statistics.

| candidate (rank-1, K=64 neurons per layer) | looped | finished within cap | with final answer | correct |
|---|---|---|---|---|
| base | 13/16 | 0/16 | 0/16 | **0/16** |
| paper's edit (their u, L15+19, x6) | 5/16 | 1/16 | 1/16 | **1/16** |
| R1 random direction, L11,15,19,23, x6 | 13/16 | 0/16 | 0/16 | **0/16** |
| A4 variance-weighted dir, L15,19, x6 | 15/16 | 0/16 | 0/16 | **0/16** |
| A6 variance-weighted dir, L3-23 (6 layers), x6 | 13/16 | 3/16 | 0/16 | **0/16** |
| A5 variance-weighted dir, L11,15,19,23, x6 | 12/16 | 4/16 | 2/16 | **2/16** |
| A3 mean-difference dir, L11,15,19,23, x6 | 2/16 | 14/16 | 0/16 | **0/16** |
| B1 variance-weighted dir, L11,15,19,23, x3 | 12/16 | 1/16 | 1/16 | **1/16** |
| B2 variance-weighted dir, L11,15,19,23, x12 | 8/16 | 14/16 | 2/16 | **2/16** |

*Validation of A5 on 40 problems:* correct **4/40**, finished within cap 7/40; base on the same 40: 0/40; paper's edit: 1/40. Fisher exact A5 vs base: p = 0.116 (suggestive, not conclusive at this n).

*Why a guard is needed:* candidate A3 "fixed" loops (12 %) only because the model stopped after 9-83 tokens (all answers wrong) - a proxy metric must be paired with accuracy and answer presence. The high dose (B2) ends thinking early in most problems without giving an answer; the two correct answers of A5/B2 both show a short loop that the edited model escapes from and then answers. Dose-response in finishing within the cap: x3 -> 6 %, x6 -> 25 %, x12 -> 88 % (but most without an answer).

**Take-away.** The mechanism (a loop direction in a few MLP neurons) is real and a rank-1 edit can remove most loops, but
(a) it unlocks capability the model already has rather than adding any, (b) its benefit is hidden behind very long reasoning
budgets on small thinking models, and (c) finding the right direction from imatrix statistics alone is hard: our best candidate (A5) was suggestive
(4/40 correct vs 0/40, p = 0.12) but not significant, and the one candidate with a dramatic effect on loops (A3) simply broke the model.

## 3b. Model selection for this laptop (Qwen3.5 family, Apache-2.0, thinking off, no tools)

| model (quant) | file size | plain decode tok/s | HumanEval greedy pass@1 | sampled pass@1 (T = 0.8) | pass@4 | with 2 repair rounds (test-error feedback) |
|---|---|---|---|---|---|---|
| Llama 3.2 3B Q4_K_M (reference) | 2.0 GB | 106.5 | not measured | - | - | - |
| Qwen3.5-0.8B Q8_0 | 0.8 GB | 219.6 | not measured | - | - | - |
| Qwen3.5-2B Q4_K_M | 1.3 GB | 150.7 | 51.8 % | 46.6 % | 66.5 % | 57.3 % |
| Qwen3.5-4B Q4_K_M | 2.7 GB | 71.8 | **82.9 %** | 79.9 % | **89.0 %** | **89.0 %** |

Notes: HumanEval problems are public (possible training-data overlap), so read this as a relative comparison. The 4B model ran out of
VRAM at 16 slots / 64k context on 8 GB (hybrid models keep a recurrent state per slot); 8 slots / 32k worked. An earlier, crashed 4B
run gave 84.1 % greedy (run-to-run about +-1.2). Diversity is fuel for best-of-N: 4 samples add +15 points for the 2B and +9 for the 4B,
and two rounds of test-feedback repair bring the 4B to its pass@4 level (89 %) with a single trajectory. For "fast and smart" on 8 GB the
4B is the pick (about 72 tok/s plain, more with speculation); the 2B is the fast option; thinking-mode runs are slow (about 30-65 tok/s per stream).

## 4. Other experiments (negative / partial)

* **Llama 3.2 3B, neuron-level LoRA ablations** (loop-specific MLP neurons, layers 15-27) screened on 79 problems:
  accuracy 26.6 % (base) vs 29-32 % (four candidates) - inside the noise (SE ≈ 5 points, run-to-run ≈ 2.5).
* **Autonomous code-writing pipeline** (model plans, writes, tests, repairs a Python package with pytest + pyflakes feedback):
  with a 3B model only ~18 % of modules passed on the first attempt and repairs rarely rescued failures, because the
  model's own tests often encoded wrong expectations. A stronger base and test-time compute would be needed.

## 5. Reproduce

```
HIZ_ROOT=<folder>   # models/, bin/llama-b12473/, data/, results/, logs/ live under it
scripts/env.sh                      # keeps temp/caches inside the folder
scripts/indir_*.sh                  # resumable downloaders with SHA-256 checks
python scripts/hiz_testi.py run q4_base,q4_ds6+ngs   # speculative decoding tests
python scripts/hiz_testi.py conc q8,q4               # concurrent requests
python scripts/degerlendir.py <model.gguf> <tag> 300 200
python scripts/kod_degerlendir.py <model.gguf> <tag>
python scripts/wsp_kes.py yaz|karsilastir ...        # imatrix-derived rank-1 edit, adapter writer
```

Requires `llama.cpp` release binaries, Python 3.10+, `pip install gguf numpy pytest pyflakes`.
Generated code is executed by the harnesses: run them in a disposable folder.

## 6. Licences / credits

Scripts: MIT. Models are not redistributed: Qwen3.5 (Apache-2.0), Llama 3.2 (Llama 3.2 Community License), DSpark drafts
by `rasyosef`. The weight-surgery idea and the edited Qwen3.5-0.8B come from the public write-up and community GGUF
conversions; we only measured and re-implemented the recipe.

Experiments were designed, run and written up by the repository owner with Claude Code (Anthropic) as the coding/analysis assistant; all numbers in this README come from the files in `results/` (single machine, single run unless a ± is given). The speed-test JSON files in `results/` keep only metrics (raw token ids and model texts were removed).
