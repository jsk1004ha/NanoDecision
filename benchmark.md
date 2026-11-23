# NanoDecision2 Benchmarks

**English** | [한국어](./benchmark.ko.md)

Evaluation of Korean and English decision tasks: classification accuracy, probability quality, robustness, throughput, and memory usage.

[Project](https://github.com/jsk1004ha/NanoDecision) · [Evaluation protocol](https://github.com/jsk1004ha/NanoDecision/blob/main/docs/BENCHMARK_PROTOCOL.md)  
**Documentation updated:** November 6, 2026

> These are archived experiment results, checked against the supplied aggregate JSON files and evaluation scripts. This documentation update does not represent a new model run or an independent reproduction.

## Results at a glance

The main comparison evaluates three models on the same **1,000 source inputs per task**. Accuracy is reported as a percentage; higher is better. Bold values are the best in each column within this comparison. Source: S1.

| Model | NSMC (K=2) | XNLI-ko (K=3) | MASSIVE-ko scenario (K=18) | AG News (K=4) | Macro average |
|---|---:|---:|---:|---:|---:|
| **NanoDecision2** | **90.20** | **56.30** | **75.30** | **94.60** | **79.10** |
| Laya-Multilingual | 67.70 | 38.50 | 42.00 | 92.30 | 60.13 |
| Julia-1 | 64.90 | 39.20 | 36.60 | 85.70 | 56.60 |

NanoDecision2 has the highest accuracy on all four tasks in this harness. Its macro accuracy is **18.98 percentage points above Laya-Multilingual** and **22.50 points above Julia-1**, calculated before rounding. This is a comparison of the recorded model/API configurations, not a claim of state-of-the-art performance or a training-budget-matched experiment.

Two qualifications matter alongside the accuracy result:

- NanoDecision2 has lower recorded throughput than both baselines. The timing boundaries differ, so those measurements are not a matched end-to-end speed comparison.
- The reported **295.59 MiB** INT8 model file is a storage measurement, not the total memory required for GPU inference.

## Contents

- [Evaluation setup](#evaluation-setup)
- [Probability quality](#probability-quality)
- [Efficiency](#efficiency)
- [Robustness](#robustness)
- [Additional NanoDecision2 evaluations](#additional-nanodecision2-evaluations)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Appendix A: Model and calibration metadata](#appendix-a-model-and-calibration-metadata)
- [Appendix B: Evidence and artifact identity](#appendix-b-evidence-and-artifact-identity)
- [Appendix C: Reporting notes](#appendix-c-reporting-notes)
- [Documentation references](#documentation-references)

## Evaluation setup

### Models and runtime

| Model | Artifact / identifier | Main comparison runtime |
|---|---|---|
| NanoDecision2 | Supplied native bundle; `jhu-clsp/mmBERT-base` backbone | INT8 bundle restored to a PyTorch model, CUDA inference |
| Laya-Multilingual | `convaiinnovations/laya-multilingual` | `laya` prediction API, CUDA |
| Julia-1 | `SupersonicLabs/Julia-1` | Local `julia` engine logits API, CUDA |

The main comparison does **not** measure NanoDecision2's native C++ CPU runtime. INT8 weight storage does not mean that the GPU path performs every operation in INT8.

| Setting | Recorded configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| OS / GPU capacity | Windows 11 x64 / 8 GB, as stated in the supplied report |
| Requested batch size | 64 |
| Execution dtype | Not fully logged or verified to be identical across models |
| Software versions | No complete Python, PyTorch, CUDA, and baseline SDK lockfile supplied |

NanoDecision2 uses temperature calibration. The baselines were not recalibrated with the same data and budget. Input formatting also follows each model's API: for example, NSMC uses NanoDecision2's `noul` mode and a two-candidate choice formulation for the baselines. Treat this as a **model-plus-adapter comparison**, not an isolation of architecture or calibration effects.

### Tasks and sampling

| Dataset | Language | Evaluated target | K | Inputs per model |
|---|---|---|---:|---:|
| NSMC | Korean | Positive / negative movie-review sentiment | 2 | 1,000 |
| XNLI-ko | Korean | Entailment / neutral / contradiction | 3 | 1,000 |
| MASSIVE-ko | Korean | Scenario / domain classification | 18 | 1,000 |
| AG News | English | World / Sports / Business / Sci-Tech | 4 | 1,000 |

The runner takes the **first eligible test rows in file order**, not a random or stratified sample. There are **4,000 source inputs and 12,000 model evaluations** across three models—not 12,000 independent questions. MASSIVE uses the `scenario` field; its score is not the standard fine-grained intent-classification result.

Macro accuracy gives each task equal weight. Because all tasks contain 1,000 inputs here, it also equals accuracy across the pooled inputs. Laya's unrounded task mean is 60.125%; the gaps are 18.975 and 22.50 percentage points. Relative error reductions are 47.6% and 51.8%, respectively, using `(baseline error − NanoDecision2 error) / baseline error`. These are not relative increases in accuracy.

## Probability quality

Lower is better for NLL, Brier, and ECE. ECE is shown on a 0–1 scale: `0.0163` corresponds to 1.63 percentage points. The following macro values are the archived `summary` values, with equal task weights. Source: S1.

| Model | NLL ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|
| **NanoDecision2** | **0.4928** | **0.2687** | **0.0344** |
| Laya-Multilingual | 1.0535 | 0.5054 | 0.0894 |
| Julia-1 | 1.5509 | 0.5858 | 0.1728 |

NanoDecision2 has the lowest macro values, but it does not lead every metric on every task: Laya has lower MASSIVE scenario ECE, **0.0538 vs. 0.0674**. The calibration-budget difference described above also applies to these comparisons.

<details>
<summary>Per-task probability metrics and definitions</summary>

| Task | Model | NLL ↓ | Brier ↓ | ECE ↓ |
|---|---|---:|---:|---:|
| NSMC | NanoDecision2 | **0.2416** | **0.1454** | **0.0163** |
| NSMC | Laya-Multilingual | 0.6019 | 0.4111 | 0.0755 |
| NSMC | Julia-1 | 0.8953 | 0.5337 | 0.2033 |
| XNLI-ko | NanoDecision2 | **0.9005** | **0.5343** | **0.0376** |
| XNLI-ko | Laya-Multilingual | 1.2829 | 0.7600 | 0.1881 |
| XNLI-ko | Julia-1 | 1.2515 | 0.7322 | 0.1866 |
| MASSIVE-ko scenario | NanoDecision2 | **0.6423** | **0.3057** | 0.0674 |
| MASSIVE-ko scenario | Laya-Multilingual | 2.0399 | 0.7265 | **0.0538** |
| MASSIVE-ko scenario | Julia-1 | 3.5946 | 0.8458 | 0.2263 |
| AG News | NanoDecision2 | **0.1868** | **0.0895** | **0.0164** |
| AG News | Laya-Multilingual | 0.2894 | 0.1241 | 0.0402 |
| AG News | Julia-1 | 0.4624 | 0.2314 | 0.0751 |

- **Accuracy:** fraction of inputs whose highest-probability candidate matches the reference label.
- **NLL:** mean negative natural logarithm of the reference-label probability; the main runner clips probabilities at `1e-12` for the logarithm.
- **Brier:** mean squared probability error, summed over all candidates per input. It is not divided by K.
- **ECE:** the absolute gap between each bin's accuracy and mean confidence, weighted by that bin's share of inputs, over **10 equal-width confidence bins** in the main comparison. The separate single-model evaluations S3 and S4 use **15 bins** and should not be combined as if they used the same ECE configuration.

Re-averaging already-rounded task values may differ in the last decimal place from the stored macro summary.

</details>

## Efficiency

### Recorded throughput

Unit: **samples/s**; requested batch size 64. These are the rates recorded by the supplied harness, not standardized latency measurements. Source: S1.

| Model | NSMC | XNLI-ko | MASSIVE-ko scenario | AG News |
|---|---:|---:|---:|---:|
| NanoDecision2 | 125.9 | 105.2 | 23.4 | 98.1 |
| Laya-Multilingual | 185.9 | 175.6 | 107.1 | 154.8 |
| Julia-1 | 208.1 | 338.1 | 215.2 | 238.0 |

**Timing is not aligned across models.** NanoDecision2 prepares and tokenizes inputs before its timer; Laya times prediction API calls; Julia's timed region includes row preparation, API execution, and probability processing. Matched warm-up, execution precision, and repeated timing runs are not established by the archive.

The recorded mean rates of 88.2 / 155.9 / 249.9 are arithmetic means of the four task rates, not total samples divided by total elapsed time. Likewise, `latency_ms_per_sample` is batch-amortized time—not single-request latency, P50, or P95.

### Model storage and CUDA memory

Unit: **MiB = 1,048,576 bytes**. The source JSON uses `_mb` field names, but the script divides by 1024². CUDA figures are PyTorch allocated-memory statistics, not total GPU usage including drivers and other processes. Source: S2.

| Runtime | Recorded model storage | CUDA allocated after loading | CUDA allocated peak, BS=1 | CUDA allocated peak, BS=64 |
|---|---:|---:|---:|---:|
| NanoDecision2 · Native CPU | 295.59 | — | — | — |
| NanoDecision2 · PyTorch CUDA | 295.59 | 1,195.1 | 1,205.4 | 1,336.4 |
| Laya-Multilingual · PyTorch CUDA | 614.01 | 1,247.2 | 1,499.0 | 1,573.4 |
| Julia-1 · PyTorch CUDA | 550.45 | 551.4 | 650.2 | 682.9 |

The NanoDecision2 difference of **10.3 MiB** (`1,205.4 − 1,195.1`) is the additional BS=1 allocation after loading, **not total VRAM**. Storage measurements are also not complete installation sizes: the script measures `model.nd2` for NanoDecision2 and aggregates cached weight files for Laya, with a constant fallback if no matching files are found. They are not identically measured distribution packages.

<details>
<summary>Host memory and workload scope</summary>

| Runtime | Post-inference process RSS (MiB) |
|---|---:|
| NanoDecision2 · Native CPU | 574.6 |
| NanoDecision2 · PyTorch CUDA | 2,256.4 |
| Laya-Multilingual · PyTorch CUDA | 2,625.6 |
| Julia-1 · PyTorch CUDA | 1,305.7 |

Despite the original key containing `peak`, these values are single process-RSS snapshots after inference, not continuously observed peak RAM. The native CPU harness also imports PyTorch, so its RSS is not a measurement of a standalone C++ executable.

The memory workload repeats short, two-candidate inputs. It does not establish requirements for long contexts, large candidate sets, concurrent requests, or compatibility with every 4 GB GPU.

</details>

## Robustness

### Candidate-order sensitivity

AG News: **300 inputs × 5 random candidate orders per model**, with seed 42; sampled orders may repeat. Consistency is the fraction of inputs with the same semantic prediction across all sampled orders, after mapping shuffled positions back to candidate identities. Source: S5.

| Model | Prediction consistency ↑ | Inputs with a prediction flip ↓ | Maximum reference-label probability range ↓ |
|---|---:|---:|---:|
| NanoDecision2 | **100.00%** | **0.00%** | **0.000000** |
| Laya-Multilingual | 95.67% | 4.33% | 0.697300 |
| Julia-1 | 90.33% | 9.67% | 0.760306 |

The flip-rate denominator is 300 inputs, not 1,500 inference calls. The probability range concerns the **reference-label probability**, not the full vector, and is stored to six decimal places. No observed change in this sample is not a proof of invariance for all inputs. A separate S4 experiment uses 500 inputs and three cyclic orders including the original order; it is not pooled with this comparison.

### Korean spelling obfuscation

This is **sentiment classification on altered text**, not an LLM safety-jailbreak benchmark. Accuracy is in percent. Source: S6.

| Condition | Inputs per model | NanoDecision2 | Laya-Multilingual | Julia-1 |
|---|---:|---:|---:|---:|
| Hand-authored obfuscated review-style examples | 50 | **86.0** | 60.0 | 60.0 |
| Clean NSMC | 1,000 | **90.0** | 69.4 | 61.6 |
| Obfuscated versions of the same NSMC inputs | 1,000 | **82.4** | 60.7 | 56.9 |
| Accuracy retention: obfuscated / clean | — | 91.56% | 87.46% | **92.37%** |

NanoDecision2 has the highest absolute obfuscated-text accuracy; Julia-1 has the highest **relative retention**. The 50 hand-authored examples are balanced, with 25 positive and 25 negative cases, and are not an independently collected real-review corpus. NSMC mutations use deterministic seeds and assume sentiment labels are preserved; independent relabeling is not documented.

The NSMC selection rule is the same as in the main comparison, but questions and evaluation settings differ, so clean scores are not same-configuration replications. These observations do not isolate a tokenizer or attention mechanism as the cause of the differences.

### Spacing removal and typo injection

NanoDecision2-only evaluation of the same **3,000 NSMC source inputs** under three conditions. This is 9,000 evaluations, not 9,000 independent source inputs. Source: S4.

| Condition | Inputs | Accuracy ↑ | NLL ↓ | Brier ↓ |
|---|---:|---:|---:|---:|
| Clean | 3,000 | 89.60% | 0.2425 | 0.1458 |
| Spaces removed | 3,000 | 88.90% | 0.2625 | 0.1582 |
| 5% typo injection | 3,000 | 88.20% | 0.2745 | 0.1658 |

The observed accuracy drops are 0.70 and 1.40 percentage points, respectively.

## Additional NanoDecision2 evaluations

### Larger classification and score-prediction runs

These experiments use different sample counts, prompts, and input representations from the main comparison. **Do not combine these scores with the baseline columns above.** ECE uses 15 bins. Sources: S3–S4.

| Task | Evaluation rows | Accuracy ↑ | NLL ↓ | ECE ↓ | Additional metric |
|---|---:|---:|---:|---:|---|
| NSMC | 9,998 | 89.36% | 0.2533 | 0.0092 | — |
| XNLI-ko | 5,010 | 82.63% | 0.4548 | 0.0167 | — |
| AG News | 2,500 | 94.36% | 0.1810 | 0.0210 | — |
| MASSIVE-ko scenario, K=18 | 2,974 | 90.28% | 0.3930 | 0.0416 | Top-3: 97.04% |
| Yelp, 1–5-star `score` | 5,000 | 68.08% | 0.7270 | 0.0285 | Within ±1 star: 97.52%; expected-score MAE: 0.4036 stars |

Yelp accuracy means an exact star-label match; normalized ordinal CRPS is 0.06036. The XNLI difference between **56.30%** in the main comparison and **82.63%** here is not a controlled improvement: both the sample size and state/question formatting change.

### Confidence-based abstention

The following values come from `fixed_threshold_diagnostics` in the NSMC 9,998-row run. Confidence is the maximum predicted candidate probability. Source: S3.

| Confidence threshold | Accepted rows | Coverage ↑ | Accuracy on accepted rows ↑ |
|---|---:|---:|---:|
| ≥ 0.70 | 8,839 | 88.41% | 93.48% |
| ≥ 0.80 | 8,001 | 80.03% | 95.53% |
| ≥ 0.90 | 6,708 | 67.09% | 97.35% |

Higher thresholds improve observed accepted-set accuracy while rejecting more inputs. These are single-model diagnostic results, not baseline comparisons or guarantees of safe automation under deployment shift. Threshold selection and final evaluation must be separated.

<details>
<summary>Small diagnostic and training-split results</summary>

- **Korean semantic contrasts (S7):** 20 correct out of 32 items, **62.5%**. The set contains 16 categories with two items each. It is too small to establish broad competence in negation, conditions, tense, or other Korean semantics; per-category 0% or 100% results are not stable ability estimates.
- **KorNLI training-split diagnostic (S4):** 20,000 rows, accuracy **89.185%**, NLL **0.3134**, ECE **0.0344**. The script reads `multinli.train.ko.tsv`. This is deliberately excluded from the held-out evaluation table and is not evidence of generalization without a training-overlap audit.

</details>

## Reproducibility

The archived metrics can be inspected, but the supplied package is **not a turnkey, fully pinned reproduction bundle**. It lacks complete baseline/runtime locks, local Julia engine code, dataset snapshots, and per-example predictions. Benchmark result JSON files also do not bind each run to a weight-file hash.

### Required artifacts

| Artifact | Purpose |
|---|---|
| NanoDecision source at the evaluation commit | Runtime and evaluation implementation |
| `model.nd2`, `tokenizer.json`, `manifest.json`, `calibration.json` in one bundle directory | Matching weights, tokenizer, metadata, and calibration |
| `nanodecision2_benchmark_suite.zip` | Archived `scripts/`, `raw_data/`, and reports |
| Exact Laya and Julia weight revisions and SDK/local engine versions | Reconstructing the baseline execution paths |
| Dataset snapshots and dependency/hardware records | Reconstructing inputs and execution conditions |

The supplied metadata file is named `manifest(1).json`; use `manifest.json` inside the runtime bundle. The weight file itself was unavailable for this documentation check; its declared hash has not been independently verified here.

### Execution entry points

1. Follow the repository's environment requirements and install from its source checkout with `python -m pip install .`. The native runtime needs a C++17 toolchain; GPU comparison dependencies and baseline SDKs must also be installed.
2. Unpack the benchmark archive. Replace the machine-specific bundle, baseline, output, and `sys.path` settings in the scripts. Pin model/data revisions and confirm file hashes before running.
3. From the unpacked benchmark-suite directory, after adapting those paths and dependencies, use the corresponding script entry points:

```bash
# Main three-model comparison
python scripts/run_open_model_comparison.py

# Separate storage / memory measurement
python scripts/measure_memory_footprint.py

# Additional NanoDecision2-only evaluations
python scripts/run_large_scale_eval.py
python scripts/run_advanced_test_suite.py

# Korean spelling-obfuscation comparison
python scripts/run_korean_evasion_eval.py
```

These are script entry points, not a claim that the original package runs unchanged on a new machine. In particular:

- The comparison conditionally restores Julia's encoder `forward` function, so the matching local implementation is required.
- The main runner infers language from question strings. XNLI and MASSIVE should use explicit Korean metadata, with the applied temperature recorded. The supplied calibration file has no matching language-specific bucket for those candidate counts; the language heuristic alone does not establish that the recorded scores change.
- `run_deep_comparative_suite.py` contains hard-coded selective-classification curves. Running it does not turn those arrays into measured results; the curves must be computed from saved labels and probabilities.

For a new controlled comparison, follow the project's [benchmark protocol](https://github.com/jsk1004ha/NanoDecision/blob/main/docs/BENCHMARK_PROTOCOL.md): log revisions, dtypes, input limits, preprocessing/timing boundaries, warm-up and repetitions, and per-example predictions, failures, truncations, and abstentions. Audit source-group overlap between training, calibration, and evaluation. Use independently held-out data for final thresholds and group-aware uncertainty estimates where appropriate. The archived suite should not be assumed to satisfy every requirement of that protocol.

## Limitations

- **Scope:** these are selected Korean/English classification and decision tasks, not a general multilingual, generative reasoning, or safety benchmark. The first-row sampling scheme limits population-level conclusions.
- **Data exposure:** dataset names and test filenames do not establish that the model was never exposed to related examples during training or calibration. No complete contamination audit is supplied; no zero-shot or out-of-distribution guarantee is claimed.
- **Uncertainty:** the main aggregate JSON does not provide all per-example predictions needed to reconstruct paired tests or source-group-aware confidence intervals for between-model differences. Differences are reported descriptively, without claims of statistical significance.
- **Deployment:** calibrated probabilities, confidence thresholds, and conformal prediction sets do not certify that an action is safe or that performance will hold after distribution shift.

Measurement-specific qualifications remain next to the affected tables. Model identity, calibration scope, and excluded claims are recorded below.

## Appendix A: Model and calibration metadata

<details>
<summary>Model configuration and training provenance</summary>

Source: M1. These are manifest records, not an independent recount of parameters or training examples.

| Field | Recorded value |
|---|---|
| Backbone | `jhu-clsp/mmBERT-base` |
| Backbone revision | `c5955035435e2bf121cde7f3c8863ef52ff35d82` |
| Encoder layers / hidden dimension / attention heads | 22 / 768 / 12 |
| Intermediate dimension | 1,152 |
| Decision dimension / decision hops | 256 / 2 |
| Tokenizer | BPE, vocabulary 256,000 |
| Weight storage | Row-wise INT8 (`int8_rowwise`) |
| Native arithmetic | FP32; optional dequantized BLAS tiles |
| Default state-input limit | 2,048 tokens |
| Question–candidate pair limit | 256 tokens |
| Architectural position capacity | 8,192 positions |
| Training objectives | Supervised NLL + Brier + ordinal CRPS |
| Recorded training / validation rows | 5,000,000 / 6,000 |

The row count does not establish five million independent source documents. Position capacity is not a measured 8K-input benchmark result. An exact parameter count has not been independently calculated for this document. The manifest's `quality_status` is `UNVERIFIED`; export success and metadata consistency are not substitutes for quality evaluation.

</details>

<details>
<summary>Temperature and conformal calibration</summary>

Source: M2. In addition to a global temperature, the file contains selected language/task/candidate-count bucket temperatures.

| Field | Recorded value |
|---|---:|
| Temperature-calibration rows | 4,000 |
| Global temperature | 1.194711 |
| NLL before calibration | 0.368392 |
| NLL after calibration | 0.362432 |
| Conformal input rows | 4,000 |
| Conformal groups | 2,720 |
| Conformal alpha | 0.10 |
| Conformal method | split group-max LAC |

The before/after NLL values describe the calibration record, not an improvement measured on a new independent test set. The conformal guarantee concerns marginal simultaneous label-set coverage for a fresh group, under exchangeability of complete groups and the same question-sampling process. It is not a guarantee of pointwise accuracy, per-domain accuracy, accuracy only on accepted answers, or action safety.

</details>

## Appendix B: Evidence and artifact identity

### Evidence index

S1–S7 paths are relative to `nanodecision2_benchmark_suite.zip`. They identify the supplied evidence; those raw files are **not embedded in this Markdown document**. M1–M3 were supplied separately.

| ID | Result / metadata file | Corresponding script | Coverage in this document |
|---|---|---|---|
| S1 | `raw_data/open_model_comparison.json` | `scripts/run_open_model_comparison.py` | Four-task model comparison and throughput |
| S2 | `raw_data/memory_benchmark_report.json` | `scripts/measure_memory_footprint.py` | Storage, CUDA allocations, post-inference RSS |
| S3 | `raw_data/large_scale_report.json` | `scripts/run_large_scale_eval.py` | Larger NSMC/XNLI/AG News runs and fixed confidence thresholds |
| S4 | `raw_data/advanced_test_report.json` | `scripts/run_advanced_test_suite.py` | Yelp, MASSIVE, KorNLI diagnostic, noise and separate permutation tests |
| S5 | `raw_data/deep_comparative_report.json` | `scripts/run_deep_comparative_suite.py` | Candidate-order comparison only; constant selective curves excluded |
| S6 | `raw_data/korean_evasion_report.json` | `scripts/run_korean_evasion_eval.py` | Korean spelling-obfuscation sentiment classification |
| S7 | `raw_data/full_report.json` | `scripts/run_tests.py` | Korean semantic-contrast item records |
| M1 | `manifest.json` | — | Model structure, training provenance, declared weight hash |
| M2 | `calibration.json` | — | Calibration parameters, guarantee scope, declared weight hash |
| M3 | `tokenizer.json` | — | Tokenizer format, vocabulary, file identity |

<details>
<summary>SHA-256 checksums and verification status</summary>

The supplied tokenizer hash matches the manifest. The manifest and calibration file declare the same model hash, but the unavailable `model.nd2` has not been rehashed or executed for this documentation update.

| Artifact | SHA-256 | Status |
|---|---|---|
| `model.nd2` | `14482c45a566e9ee9bb194541d59dcd7f56968d32c1e7f32776b190c11257bbb` | Declared in metadata; weight bytes not independently checked |
| `tokenizer.json` | `197d4cc5406ee12cc50c8b5511f2393cc32d9db321545979ce041c1199178356` | Supplied file hashed; matches manifest |
| `calibration.json` | `cec6e97dc356b54c7cc8e163f3ddb77f77da92a5c8bf152c127b8b28fb240ae0` | Hash of supplied file |
| `manifest.json` | `d005c079efd51313256f367cdd02fa0390fd1cc76a09c463a32908d0647e162b` | Hash of supplied `manifest(1).json` bytes |
| Benchmark-suite ZIP | `c1788e6eaf4fda2f1500e1b8da8ddf8b9b8b75113f51a308914e6896c44c2680` | Hash of supplied archive |

</details>

## Appendix C: Reporting notes

<details>
<summary>Claims corrected or excluded after checking the archived scripts</summary>

| Original claim / presentation | Reporting decision |
|---|---|
| 89.5% automation at 95% accuracy; three-model risk–coverage curves | Excluded as empirical evidence: `run_selective_classification_test()` assigns constant arrays and coverage strings instead of calculating them from predictions. |
| 70,087 independent evaluation questions | Not used: source inputs, repeated model runs, mutations, and permutations have different counting units. |
| KorNLI test set, 20,000 rows | Labeled as a training-split diagnostic because the script reads `multinli.train.ko.tsv`. |
| Korean semantic diagnostic: 55 items, 65.5% | Replaced by the item-level record: 20/32 correct, 62.5%. |
| 10.3 MiB total VRAM; 25× memory efficiency | Not supported: 10.3 MiB is the incremental BS=1 allocation after model loading. |
| 574.6 MiB peak native host RAM | Relabeled as post-inference process RSS in a Python harness that also imports PyTorch. |
| Mathematically zero probability change for all candidate orders | Limited to the sampled inputs/orders and the recorded reference-label probability range. |
| Candidate-scaling result of 91.9% at K=18 | Excluded from headline results: its separate result JSON is absent from the archive. |
| Structural superiority inferred from token fragmentation | Excluded: token-count APIs and special-token inclusion are not aligned across models. |
| Exact 64–2,048-token latency scaling / validated 8K performance | Not established: string repetition and character slicing do not verify actual tokenizer lengths. |
| 89.5 ms average latency proves an operational SLA | Excluded: the matching independent latency record and deployment-load conditions are unavailable. |

The original composite figures may include these excluded claims. They are not reused in this document; any future figures should be regenerated from the reviewed results with the same qualifications.

</details>

## Documentation references

The organization of this document draws on the following official open-model documentation, reviewed on November 6, 2026. They are **format references only**, not additional baselines or sources of NanoDecision2 scores.

- [Qwen3-Embedding: Evaluation](https://github.com/QwenLM/Qwen3-Embedding#evaluation) and [evaluation README](https://github.com/QwenLM/Qwen3-Embedding/blob/main/evaluation/README.md): benchmark-specific comparison tables, notes, and separate reproduction instructions.
- [ModernBERT-base model card](https://huggingface.co/answerdotai/ModernBERT-base): concise model context, grouped evaluation tables, and a dedicated limitations section.
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3): separation of evaluation sources and disclosure of task-specific training and comparison conditions.

---

Both language editions report the same archived experiments, values, and qualifications. Only the prose is translated.
