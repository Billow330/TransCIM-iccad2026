# TransCIM Project Handoff — Guide for AI Agents

**Purpose**: This README helps the next AI agent (or human) quickly understand and continue the TransCIM project after migration to a new server. It consolidates project context, structure, and commands.

---

## 交接文件在哪里 / Where to Find Handoff Files

| 文件 | 路径（以 NeuroSimV1.5 为根） | 说明 |
|------|------------------------------|------|
| 本交接指南 | `HANDOFF_README.md` | 项目结构、脚本、数据流、复现命令 |
| 完整聊天记录 | `docs/chat_history_handoff.md` | 此前全部 AI 对话（约 12 万行） |
| 论文 + 图表数据 | `results/analysis/paper_complete_with_figures.md` | Abstract + Appendix（数据源、脚本、CSV） |
| 论文正文 | `results/analysis/paper_complete_merged.md` | 完整 Sections 2–7 |

**AI 接手时**：先读 `HANDOFF_README.md`（本文件），再按需查 `docs/chat_history_handoff.md` 和 `results/analysis/paper_complete_with_figures.md`。

---

## 1. Project Overview

- **Project**: TransCIM — A Universal Compute-in-Memory Optimization Framework for Transformer Architectures
- **Target venue**: ICCAD 2026
- **Main codebase**: `NeuroSimV1.5/` (this folder)

TransCIM provides:
1. **Architecture-aware mapping**: ViT (block-wise) vs. Swin (window-aware)
2. **Layer-adaptive precision presets**: standard / conservative / balanced / aggressive
3. **Attention optimizations**: QK^T tiling, softmax approximation

Models: ViT-Base/16, Swin-Tiny on ImageNet-1K.

---

## 2. Directory Structure

```
NeuroSimV1.5/               # 项目根目录
├── README.md               # 入口，含「AI 接手必读」
├── HANDOFF_README.md       # 本文件，交接总览
├── inference.py
├── quantize.py
├── dataset.py
├── models/                 # ViT, Swin, ResNet, VGG
├── NeuroSIM/               # CIM simulator (C++ backend)
├── outputs/
│   └── hw_metrics/         # summary.csv, per_layer.csv
├── results/
│   ├── analysis/           # 论文、CSV、图表
│   └── figures/            # PNGs
├── docs/
│   ├── chat_history_handoff.md   # 完整聊天记录
│   └── ppa_params.yaml
├── scripts/
├── configs/
└── pytorch-quantization/
```

---

## 3. Key Files (Must Know)

| File | Role |
|------|------|
| `results/analysis/paper_complete_merged.md` | Full paper body (Sections 2–7) |
| `results/analysis/paper_complete_with_figures.md` | Abstract + Appendix (figure data, CSVs, scripts) |
| `outputs/hw_metrics/summary.csv` | Per-config PPA (latency_ms, energy_pj, traffic_scale) |
| `outputs/hw_metrics/per_layer.csv` | Per-layer trace (ic_dynamic_energy_pj, etc.) |
| `results/analysis/e2e_latency_bw_sweep.csv` | BW sweep (Figure 3) |
| `results/analysis/spill_buffer_sweep.csv` | Buffer spill model (Figure 4) |
| `results/analysis/kernel_ppa_scaled.csv` | Bitwidth PPA (Table 2-s) |
| `results/analysis/validation_50k.json` | 50k ImageNet validation |
| `docs/ppa_params.yaml` | e_NoC, e_DRAM, BW, etc. |

---

## 4. Data Flow and Scripts

From `NeuroSimV1.5/` root:

| Script | Input | Output |
|--------|-------|--------|
| `compute_e2e_latency_bw_sweep.py` | summary.csv, per_layer.csv, ppa_params.yaml | e2e_latency_bw_sweep.csv, fig_e2e_latency_vs_bw.png |
| `compute_spill_buffer_sweep.py` | e2e_latency_bw_sweep.csv, summary.csv | spill_buffer_sweep.csv, fig_energy_total_vs_buffer.png |
| `compute_end_to_end_ppa.py` | per_layer.csv, summary.csv | ppa_end_to_end.csv |
| `scale_kernel_ppa_by_bitwidth.py` | summary.csv | kernel_ppa_scaled.csv, table2_kernel_ppa_scaled.md |
| `run_50k_validation.py` | ImageNet val | validation_50k.json |
| `run_hw_metrics.py` | Models, NeuroSim | summary.csv, per_layer.csv |

---

## 5. Context from Previous AI Sessions

- Buffer-limited spill scenario (8KB, γ=0.5): DRAM 8.38% / 9.33%, NoC 0.68%
- BW-sweep (Figure 9): 15.00% / 16.70% e2e latency ↓ at BW=32 GB/s
- Table 2-s: baseline 0.920 mJ, full_opt 0.830 mJ (9.80%↓)
- `inference.py` `--hardware 0` skips NeuroSim for accuracy-only
- `dataset.py` StratifiedRandomSampler uses `dataset.targets` (no image decode at init)
- Conventions: 15.00%, 16.70%, 9.80%, 8.38%, 9.33%, 0.68%; 0.920/0.830 mJ; no "accuracy improvement" claim

---

## 6. Quick Start for New AI

1. Read `HANDOFF_README.md` (this file)
2. Read `results/analysis/paper_complete_with_figures.md` Appendix
3. Check `outputs/hw_metrics/summary.csv`, `results/analysis/*.csv`
4. Search for `compute_*.py`, `run_*.py` if scripts are missing
5. Pipeline: `run_hw_metrics` → `compute_*` → update paper

---

## 7. Open Items

- Full 50k quantized validation
- Verify scripts exist and run
- Ensure `results/figures/` has PNGs
- Merge paper files into LaTeX if needed

---

*Last updated: 2026 (project handoff).*
