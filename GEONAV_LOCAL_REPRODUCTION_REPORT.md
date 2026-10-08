# GeoNav 本地复现报告

## 结果状态

官方 `test_unseen` 全量 **5,311/5,311** episodes 均有非空轨迹 JSON；episode 指标文件含 5,311 行、5,311 个唯一 episode ID，序号覆盖 1–5,311。初次评测有 1 条 `IndexError`，runner 生成不完整清单后，断点续跑成功补齐。最终清单：`results/geonav_deepseek_flash_full_corrected/full_test_unseen_manifest.json`；逐 episode 指标：`results/geonav_deepseek_flash_full_corrected/geonav_origin/test_unseen/episode_metrics_test_unseen.jsonl`。初次失败记录保存在 `results/geonav_deepseek_flash_full_corrected/initial_attempt_incomplete_test_unseen.json`。

| Metric | Local, 5,311 episodes | Paper test unseen (mean ± std, 5 runs) | Local − paper |
|---|---:|---:|---:|
| NE (m, lower is better) | 68.01 | 73.5 ± 1.2 | -5.49 m |
| SR (%) | 26.66 | 25.9 ± 1.1 | +0.76 pp |
| OSR (%) | 43.55 | 41.6 ± 1.8 | +1.95 pp |
| SPL (%) | 20.38 | 16.0 ± 0.9 | +4.38 pp |

Local metric values were independently recomputed as arithmetic means from the per-episode JSONL. NE 68.01 m, SR 26.66%, OSR 43.55%, SPL 20.38%.

## 设置与差异

- Source: official repository [`Xhtshr/geonav-official`](https://github.com/Xhtshr/geonav-official), checkout commit `925fe1a76c096d89d35a9736f4a98efe1c9c92ce`. Local source contains run-enabling fixes; it was not pushed upstream.
- Official README evaluation settings followed: altitude 50 m, segmentation mask enabled, detector threshold 0.20, max timestep 20, split `test_unseen`. Machine-available GSAM cache is `full_scan_(100, 240, 410).npz`, so the run uses map size 240 and map extent 410 m with the official map cache.
- Paper implementation uses GPT-4o; this local run substitutes DeepSeek Flash via an OpenAI-compatible API. Thus this is a full-split, model-substituted reproduction, **not an exact model reproduction**. Paper numbers are mean ± standard deviation over five runs, while local numbers are one run; deltas are descriptive, not a matched statistical comparison.
- Runtime emitted a PyTorch warning that the installed build does not support the RTX PRO 6000 Blackwell `sm_120` architecture. Reported navigation results still completed; no claim of GPU-accelerated detector inference is made because the map cache supplies GSAM observations.
- Paper reports 20 m success radius; local runner uses the same. Paper’s MLLM reasoning invocation interval is 10 action steps; the repo’s evaluation cap is 20 timesteps. The paper reports test-unseen GeoNav NE 73.5 ± 1.2 m, SR 25.9 ± 1.1%, OSR 41.6 ± 1.8%, SPL 16.0 ± 0.9%. [Paper, Table 1 and Appendix E](https://arxiv.org/html/2504.09587)

## 20-epoch Heatmap run

HETT Heatmap 20 epochs (zero-based 0–19) metrics and sample-coverage record are in [`epoch_metrics.csv`](/home/rental/20260922_1/Workspace/hett-heatmap-system-opt/artifacts/heatmap_20epoch_complete/epoch_metrics.csv) and its `README.md`. Checkpoints `latest` and `best_val_unseen` are retained under `hett-heatmap-system-opt/checkpoints/heatmap_densebelief28_a9d95e3_b8_adam_20e_resume20_epochfix/`; resume manifest records checksums and complete per-epoch sample counts.

## Metric definitions

NE is mean final distance to target (m); SR is final stop within 20 m; OSR is any point along trajectory within 20 m; SPL is success weighted by path length. Source report: [GeoNav paper](https://arxiv.org/html/2504.09587).
