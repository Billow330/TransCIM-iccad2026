We further introduce a **bitwidth-aware post-processing model** that scales the NeuroSim kernel PPA
without modifying the C++ backend. Starting from the base kernel latency/energy reported in
`outputs/hw_metrics/summary.csv`, we treat the **standard** preset (ConfigManager `PrecisionConfig`
with ADC bits `b_ref = 7`) as the reference and derive an *effective* ADC/DAC bitwidth for each precision preset.
For each preset, we conservatively use the ADC bits as a proxy for the most bit-sensitive operators
(Q/K/LN), and the input bits as the effective DAC bits; this is documented as a conservative upper
bound so reviewers can audit the assumptions.

To capture implementation uncertainty we sweep **energy shares** between ADC/DAC and constant logic:
ADC share ∈ {0.3, 0.5, 0.7}, DAC share ∈ {0.1, 0.2, 0.3}, with the constant share defined as
1 − adc_share − dac_share (invalid combinations are skipped). For each point in this grid we evaluate
two scaling forms: (1) a **linear** model, where energy scales proportional to bitwidth
(s_adc = b_adc / b_ref, s_dac = b_dac / b_ref), and (2) an **exponential-style** upper bound
(s_adc = 2^{b_adc} / 2^{b_ref}, likewise for DAC). The scaled kernel energy is then
E_scaled = E_base · (const_share + adc_share·s_adc + dac_share·s_dac).

Latency is modeled with a simple, explainable sensitivity term
T_scaled = T_base · (1 + k · (b_adc − b_ref) / b_ref), with k ∈ {0.1, 0.2, 0.3} capturing how strongly
ADC resolution impacts timing. Scaled power and energy efficiency follow directly as
P_scaled = E_scaled / T_scaled, and TOPS/W_scaled = throughput_TOPS / P_scaled (when throughput is available).
The full grid of scaled values for all configurations is recorded in
`results/analysis/kernel_ppa_scaled.csv`, while Table 2-s summarizes representative conservative /
nominal / aggressive settings for the key Swin and ViT configurations.

This bitwidth-aware scaling thus **promotes the precision preset from a pure accuracy knob to an
explicit PPA knob**: lower-bit presets (e.g., `balanced` or `aggressive`) consistently reduce the
scaled kernel energy across the nominal sensitivity setting, while conservative higher-bit presets
move the design toward a more accuracy-biased, energy-heavier operating point. Importantly, this
analysis is fully auditable: all assumptions (b_ref, ADC/DAC shares, scaling forms, and k) are
enumerated in the CSV, and no changes are made to the underlying NeuroSim C++ implementation.
