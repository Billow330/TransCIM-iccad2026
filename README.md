# TransCIM — Universal CIM Optimization Framework for Transformers

**Target**: ICCAD 2026  
**Project**: ViT/Swin on NeuroSim V1.5, ImageNet-1K

---

## AI 接手必读 / For AI Agents Taking Over This Project

**如果你是新接手的 AI，请先阅读以下文件，再开始编码或实验：**

| 优先级 | 文件路径 | 说明 |
|--------|----------|------|
| 1 | `HANDOFF_README.md` | 项目交接总览：结构、脚本、数据流、复现命令、已完成的修改 |
| 2 | `docs/chat_history_handoff.md` | 完整聊天记录（约 12 万行），包含此前所有对话与决策 |
| 3 | `results/analysis/paper_complete_with_figures.md` | 论文稿 + 图表数据附录（数据源、脚本、CSV 路径） |

**路径说明**（以 NeuroSimV1.5 为项目根目录）：
- 交接文档：`NeuroSimV1.5/HANDOFF_README.md`
- 聊天记录：`NeuroSimV1.5/docs/chat_history_handoff.md`
- 论文与图表：`NeuroSimV1.5/results/analysis/paper_complete_with_figures.md`、`paper_complete_merged.md`

**建议阅读顺序**：HANDOFF_README.md → paper_complete_with_figures.md（Appendix）→ 需要时查阅 chat_history_handoff.md

---

## 项目简介

TransCIM 提供：
1. 架构感知映射（ViT block-wise / Swin window-aware）
2. 分层混合精度预设（standard / conservative / balanced / aggressive）
3. 注意力优化（QK^T 分块、softmax 近似）

评估模型：ViT-Base/16、Swin-Tiny，ImageNet-1K。

更多说明见 `HANDOFF_README.md`。
