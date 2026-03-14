We additionally study a **buffer-limited spill model** to highlight how TransCIM's
traffic reduction can amplify end-to-end energy savings when on-chip buffer capacity
is small. Starting from the NoC bytes per configuration in
`results/analysis/e2e_latency_bw_sweep.csv` (alpha=1.0) and the kernel energy from
`outputs/hw_metrics/summary.csv`, we estimate intermediate buffer-resident bytes as
a fraction $\gamma$ of the NoC traffic, with $\gamma \in \{0.3, 0.5, 0.7\}$.

For a given buffer size $B$ (4/8/16/32 KB), we define a conservative lower-bound spill
model: $\text{spill\_bytes} = \max(0, \gamma\,\text{bytes}_\mathrm{NoC} - B)$.
These spill bytes are then charged either to the NoC (`spill_to_noc`,
$E_\mathrm{spill} = \text{spill\_bytes} \cdot e_\mathrm{NoC}$) or to off-chip DRAM
(`spill_to_dram`, $E_\mathrm{spill} = \text{spill\_bytes} \cdot e_\mathrm{DRAM}$), using
the energy-per-byte parameters from `docs/ppa_params.yaml` (default
$e_\mathrm{NoC}=1.0$ pJ/byte, $e_\mathrm{DRAM}=30$ pJ/byte). The total energy is then
approximated as $E_\mathrm{total} = E_\mathrm{kernel} + E_\mathrm{spill}$, where
$E_\mathrm{kernel}$ is the baseline kernel energy reported in Table 2.

Under this buffer-limited model, TransCIM's traffic reduction substantially reduces
spill in the **Swin-T** configuration. For example, at **8 KB** buffer and
$\gamma = 0.5$ with spills charged to DRAM, the mapping-only configuration reduces
total energy by approximately **8.4%** vs. the baseline, and the
full-opt configuration achieves a reduction of approximately **9.3%**.
In the worst-case spill-to-DRAM scenario, the largest observed total-energy
reduction is **10.7%** for `full_opt_swin` at buffer=32 KB and
$\gamma = 0.7$.

This analysis complements the compute-dominant on-chip results in Table 2 and Table 2':
when buffers are large and DRAM is disabled, total energy savings are on the order of
~1% (kernel-dominated); however, under buffer-limited or spill-heavy scenarios,
the same traffic reduction can translate into **multi-percent** end-to-end energy
savings due to avoided NoC/DRAM spill traffic.
