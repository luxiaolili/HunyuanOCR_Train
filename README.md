# HunyuanOCR Inference & Fine-tuning Project

本项目基于腾讯混元团队开源的 [HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR) 构建，支持混元视觉语言模型（VLM）的文本检测与识别任务推理，以及后续基于 verl 的后训练（fine-tuning）。

## 📋 项目概述

HunyuanOCR 是腾讯推出的多模态 OCR 模型，具备强大的图文理解与文本识别能力，支持复杂场景下的文字检测、识别及结构化输出。本项目提供SFT训练，后续提供基于verl的后训练代码

## 🛠️ 环境安装

### 系统要求
- **操作系统**: Linux
- **Python**: 3.12+ (推荐并测试版本)
- **CUDA**: 12.9
- **PyTorch**: 2.7.1
- **GPU**: 支持 CUDA 的 NVIDIA GPU
- **显存**: ≥20GB (用于 vLLM)
- **磁盘空间**: ≥6GB

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/luxiaolili/HunyuanOCR_Train.git
   cd HunyuanOCR_Train
   ```

2. **数据集**
   train.jsonl, test.jsonl
   ```
   格式：
   {"image": "xxx.png", "prompt":"提取图中的文字", "answer":"训练OCR数据"}
   ```
3. **运行**
   
   ```
   run.sh
   ```

5. **问题**
   1. HunYuanOCR对System的prompt敏感
   2. NER任务需要SFT提升
   
6. **todo**
   - [x] SFT训练
   - [ ] Verl后训练


