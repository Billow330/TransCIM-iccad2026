# 2. Background & Related Work (Draft — TransCIM, ICCAD 2026)

## 2.1 Transformer Architecture Diversity

Transformers have evolved into multiple families, each with distinct computational patterns that pose different challenges for hardware acceleration.

**Vision Transformer (ViT)** [1] applies self-attention globally across all image patches. Given N patches, the attention mechanism computes QK^T over all N×N pairs, resulting in O(N²) complexity. This global attention pattern requires handling large matrices and extensive data movement, making it challenging for CIM accelerators with limited on-chip memory.

**Swin Transformer** [2] introduces a hierarchical architecture with window-based self-attention. Instead of global attention, Swin partitions patches into fixed-size windows (e.g., 7×7) and computes attention within each window, reducing complexity to O(W²) where W is the window size. However, Swin introduces additional challenges: shifted windows require data movement between windows, and the hierarchical structure (multiple stages with different resolutions) demands adaptive mapping strategies.

**BERT** [3] and other sequence Transformers use bidirectional self-attention over sequences of length L, also resulting in O(L²) complexity. Long sequences exacerbate the softmax computation bottleneck, and the sequential nature of language tasks adds temporal dependencies that affect data flow.

These architectural differences mean that a **one-size-fits-all** CIM mapping strategy is suboptimal. ViT benefits from strategies that exploit patch parallelism and handle large matrices efficiently; Swin requires window-aware mapping and efficient inter-window data movement; BERT needs sequence-aware optimizations and softmax acceleration.

## 2.2 Compute-in-Memory Accelerators

CIM accelerators perform matrix-vector or matrix-matrix multiplications directly within memory arrays (e.g., resistive RAM, SRAM), eliminating the need to move data between memory and compute units. This reduces energy consumption and can improve throughput for memory-bound workloads.

**NeuroSim** [6] is a widely-used simulation framework for CIM accelerators, modeling power, latency, and area for various CIM architectures. It supports different mapping strategies (conventional vs. architecture-aware) and quantization schemes. However, NeuroSim's default mapping is **conventional** (not architecture-aware), and it does not provide a unified framework for optimizing across multiple Transformer architectures.

**Mapping strategies** are crucial for CIM performance. Conventional mapping treats all layers uniformly, which may not exploit architecture-specific patterns (e.g., window structure in Swin). Architecture-aware mapping adapts to the model structure, potentially improving data reuse, reducing data movement, and better utilizing CIM array parallelism.

## 2.3 Related Work

**Transformer CIM Accelerators.** CIMFormer [4] proposes a systolic CIM array design for Transformers with token-pruning-aware attention. However, it targets a specific Transformer variant and does not generalize to other architectures like Swin. HASTILY [5] introduces unified compute and lookup modules for softmax acceleration, focusing on softmax approximation but not addressing architecture diversity or unified mapping strategies.

**Limitations of existing work.** Most Transformer CIM accelerators are **architecture-specific**: they are designed for one model (e.g., ViT or BERT) and cannot be directly applied to others without redesign. They also tend to optimize **one aspect** (e.g., softmax, or attention computation) rather than providing a **comprehensive framework** that combines mapping, precision, and attention optimizations. This lack of generality limits their applicability and requires separate designs for each Transformer family.

**Mixed-precision quantization** has been explored for CNNs and some Transformers, but existing approaches are often model-specific or require manual tuning per architecture. There is a need for **universal precision presets** that can be applied across architectures while respecting layer-specific sensitivity.

**Attention mechanism optimization** has been studied extensively (e.g., sparse attention, linear attention), but most work focuses on algorithmic improvements rather than hardware-aware optimizations for CIM. Generic CIM-friendly attention optimizations (e.g., QK^T computation strategies, softmax approximations) that work across architectures are underexplored.

**Gap.** To our knowledge, there is **no existing framework** that (1) supports multiple Transformer architectures (ViT, Swin, BERT) in a unified pipeline, (2) automatically adapts mapping strategies based on architecture detection, and (3) combines architecture-aware mapping, layer-adaptive precision, and attention optimizations in a single framework. TransCIM addresses this gap.

---

*[Note: Add citations [1]-[6] as appropriate.]*
