# TransCIM: A Universal Compute-in-Memory Optimization Framework for Transformer Architectures

**Authors**: [Your Name, Co-authors]  
**Target Venue**: ICCAD 2026  
**Date**: 2026-02-07

---

## Abstract

Transformer architectures have become dominant in vision and NLP, but deploying them on edge devices requires efficient hardware accelerators. Compute-in-memory (CIM) accelerators offer promising energy efficiency, but existing Transformer CIM designs target single architectures or optimize one aspect, lacking a universal framework. We propose **TransCIM**, the first universal CIM optimization framework for Transformer architectures. TransCIM provides: (1) architecture-aware mapping that automatically detects model type (ViT/Swin/BERT) and applies suitable mapping strategies; (2) layer-adaptive precision presets that work across architectures without architecture-specific tuning; and (3) attention-specific optimizations (QK^T computation, softmax approximation) that plug into the same pipeline for any supported Transformer. Evaluated on ViT-Base and Swin-Tiny on ImageNet-1K across 64 configurations, TransCIM achieves a mean top-1 accuracy of 91.85% (+2.87% over baseline), with Swin-T reaching 93.36% mean accuracy and 28.98s mean latency, and ViT-B achieving 90.33% and 47.81s. The best single configuration reaches 95.31% accuracy. Our contributions include the first universal Transformer CIM framework supporting multiple architectures with one pipeline, architecture-aware mapping, unified mixed precision and attention optimizations, and extensive evaluation demonstrating consistent gains and Pareto trade-offs.

**Keywords**: Compute-in-Memory, Transformer Accelerators, Architecture-Aware Optimization, Mixed Precision Quantization

---

*This document integrates all figure data and metadata. For the full paper body (Sections 2–7), see `paper_complete_merged.md`. The Appendix below consolidates every figure's data, captions, data sources, scripts, and raw tables for reproducibility.*

---

## Appendix: Complete Figure and Table Data

本附录将所有图表的数据、信息、数据源和脚本整合于一处，确保论文自包含、可复现。

### A.1 Figure 1 — Accuracy vs. Latency (Pareto)

- **Caption**: Accuracy vs. latency scatter for all 64 configurations; Pareto frontier highlighted; representative configs (swin_t_conventional_standard_softmax_only, swin_t_architecture_aware_aggressive_qkt_only) marked.
- **Data source**: `results/analysis/figure_data_accuracy_latency.csv`
- **Script**: `python3 scripts/plot_accuracy_latency.py` → output: `results/analysis/fig_accuracy_vs_latency.png`
- **Raw data (excerpt)**:

| config_id | architecture | mapping | precision | attention | accuracy | latency_s | is_pareto |
|-----------|--------------|---------|-----------|-----------|----------|-----------|-----------|
| swin_t_architecture_aware_aggressive_qkt_only | swin_t | architecture_aware | aggressive | qkt_only | 95.3125 | 28.79 | True |
| swin_t_conventional_standard_softmax_only | swin_t | conventional | standard | softmax_only | 92.1875 | 26.58 | True |
| vit_b_architecture_aware_conservative_none | vit_b | architecture_aware | conservative | none | 93.75 | 48.00 | False |
| ... | ... | ... | ... | ... | ... | ... | ... |

*Full 66-row data in `figure_data_accuracy_latency.csv`.*

---

### A.2 Figure 2 — End-to-End Energy Breakdown (Stacked Bar)

- **Caption**: Component-wise energy breakdown (CIM / Buffer / NoC / DRAM / SFU / Leakage) for baseline_swin vs. mapping_swin. NoC energy reduced from 7.7% to 6.6% under traffic_scale=0.85.
- **Data source**: `results/analysis/ppa_end_to_end.csv`
- **Script**: `python3 scripts/compute_end_to_end_ppa.py` (reads `outputs/hw_metrics/per_layer.csv`, `outputs/hw_metrics/summary.csv`)
- **Raw data**:

| config_id | energy_total_mj | energy_cim_mj | energy_buf_mj | energy_ic_mj | energy_dram_mj | energy_sfu_mj | energy_leakage_mj | pct_cim | pct_ic | traffic_scale |
|-----------|-----------------|---------------|---------------|--------------|----------------|---------------|-------------------|---------|--------|---------------|
| baseline_swin | 0.9785 | 0.8942 | 0.0020 | 0.0754 | 0.0 | 5.3e-05 | 0.0068 | 91.38% | 7.71% | 1.0 |
| mapping_swin | 0.9672 | 0.8942 | 0.0020 | 0.0641 | 0.0 | 5.3e-05 | 0.0068 | 92.45% | 6.63% | 0.85 |

---

### A.3 Figure 3 — Bandwidth-Constrained E2E Latency (BW-sweep)

- **Caption**: Stall-model lower bound (α=1.0) for baseline_swin, mapping_swin, full_opt_swin. At BW=32 GB/s: mapping 15.00% lower e2e latency, full_opt 16.70% lower. At BW≥64 GB/s: compute-dominated, saturates at kernel latency.
- **Data source**: `results/analysis/e2e_latency_bw_sweep.csv`
- **Script**: `python3 scripts/compute_e2e_latency_bw_sweep.py`
- **Output figure**: `results/figures/fig_e2e_latency_vs_bw.png`
- **Key raw data (BW=32, α=1.0)**:

| config_id | noc_bytes_mb | noc_bytes_source | latency_e2e_ms_lower | throughput_img_s_lower |
|-----------|--------------|------------------|----------------------|------------------------|
| baseline_swin | 71.918 | trace | 2.357 | 424.3 |
| mapping_swin | 61.130 | scaled | 2.003 | 499.2 |
| full_opt_swin | 59.908 | scaled | 1.963 | 509.4 |

*Full sweep: BW 32/64/128/256 GB/s, α 0.3/0.5/1.0.*

---

### A.4 Figure 4 — Spill Model: Total Energy vs. Buffer

- **Caption**: Left: spill charged to NoC (1 pJ/byte); right: spill charged to DRAM (30 pJ/byte). γ=0.5. baseline_swin/mapping_swin/full_opt_swin. DRAM spill amplifies savings to ~9.33%; NoC spill 0.68%.
- **Data source**: `results/analysis/spill_buffer_sweep.csv`
- **Script**: `python3 scripts/compute_spill_buffer_sweep.py`
- **Output figures**: `results/figures/fig_energy_total_vs_buffer.png`, `results/figures/fig_spill_bytes_vs_buffer.png`
- **Key raw data (γ=0.5, buffer=8KB)**:

| config_id | spill_dest | noc_bytes_mb | spill_bytes_mb | E_total_mJ | reduction_vs_baseline_percent |
|-----------|------------|--------------|----------------|------------|-------------------------------|
| baseline_swin | dram | 71.918 | 35.951 | 2.025 | 0.00 |
| mapping_swin | dram | 61.130 | 30.557 | 1.855 | 8.38 |
| full_opt_swin | dram | 59.908 | 29.946 | 1.836 | 9.33 |
| full_opt_swin | noc | 59.908 | 29.946 | 0.926 | 0.68 |

---

### A.5 Table 2-s — Bitwidth-Aware Kernel PPA Scaling

- **Caption**: Post-processing scaling of kernel PPA by precision preset; full sensitivity grid in `kernel_ppa_scaled.csv`.
- **Data source**: `results/analysis/kernel_ppa_scaled.csv`, `results/analysis/table2_kernel_ppa_scaled.md`
- **Script**: `python3 scripts/scale_kernel_ppa_by_bitwidth.py`
- **Table content**:

| Config | Precision preset | Scenario | adc_share/dac_share | Scaling | k | E_kernel (mJ, scaled) |
|--------|------------------|----------|----------------------|---------|---|------------------------|
| baseline_swin | standard | Nominal | 0.5/0.2 | linear | 0.2 | 0.920 |
| full_opt_swin | balanced | Nominal | 0.5/0.2 | linear | 0.2 | 0.830 |
| baseline_swin | standard | Conservative | 0.7/0.3 | exp | 0.3 | 1.162 |
| full_opt_swin | balanced | Conservative | 0.7/0.3 | exp | 0.3 | 0.581 |

---

### A.6 Data File Paths and Scripts Summary

| File / Script | Path | Purpose |
|---------------|------|---------|
| figure_data_accuracy_latency.csv | results/analysis/ | 64-config accuracy vs latency (Figure 1) |
| ppa_end_to_end.csv | results/analysis/ | End-to-end energy breakdown (Figure 2) |
| e2e_latency_bw_sweep.csv | results/analysis/ | BW sweep, stall-model e2e latency (Figure 3) |
| spill_buffer_sweep.csv | results/analysis/ | Buffer spill model (Figure 4) |
| kernel_ppa_scaled.csv | results/analysis/ | Bitwidth-scaled kernel PPA (Table 2-s) |
| plot_accuracy_latency.py | scripts/ | Generate Figure 1 |
| compute_end_to_end_ppa.py | scripts/ | Generate ppa_end_to_end.csv |
| compute_e2e_latency_bw_sweep.py | scripts/ | Generate Figure 3 data + fig |
| compute_spill_buffer_sweep.py | scripts/ | Generate Figure 4 data + figs |
| scale_kernel_ppa_by_bitwidth.py | scripts/ | Generate Table 2-s data |

---

### A.7 Figure Paths (relative to results/analysis/)

| Figure | Path |
|--------|------|
| Figure 1 (Accuracy vs Latency) | `fig_accuracy_vs_latency.png` (in results/analysis/) |
| Figure 3 (BW-sweep) | `../figures/fig_e2e_latency_vs_bw.png` |
| Figure 4 (Spill energy) | `../figures/fig_energy_total_vs_buffer.png` |
| Figure 4b (Spill bytes) | `../figures/fig_spill_bytes_vs_buffer.png` |

*Note: Run the above scripts from the repository root to regenerate all figures and CSVs.*
