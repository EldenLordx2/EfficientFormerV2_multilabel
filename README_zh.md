# EfficientFormer Multi-Label Classification

本项目基于 **EfficientFormer / EfficientFormerV2** 官方实现，针对 **多标签图像分类（Multi-Label Classification）** 场景进行了工程化改造，支持：

* 使用 **TXT 文件** 描述数据集（不依赖 ImageFolder 目录结构）
* **多标签 0/1 向量训练**
* **仅包含图片路径的推理（无标签）**
* 推理结果自动输出：

  * `图片路径 + 0,1,0,...`
  * `图片路径 + 标签名1,标签名2,...`

适用于 **内容审核 / 多属性识别 / 工业多标签分类** 等任务。

---

## 📌 模型来源

本项目模型来自以下论文工作：

* **EfficientFormer** – Vision Transformers at MobileNet Speed (NeurIPS 2022)
* **EfficientFormerV2** – Rethinking Vision Transformers for MobileNet Size and Speed (ICCV 2023)

官方仓库提供了高效的 Vision Transformer 结构，本项目在其基础上**仅改造数据读取、损失函数和推理流程**，模型结构保持不变。
原始项目说明见官方 README
https://github.com/snap-research/EfficientFormer
---

## 🧩 项目特性

* ✅ 多标签分类（`BCEWithLogitsLoss`）
* ✅ 支持任意标签数（由 `label.txt` 自动推断）
* ✅ TXT 数据集格式，灵活对接已有系统
* ✅ 分布式训练 / 推理（DDP）
* ✅ 推理阶段无需标签文件
* ✅ 自动跳过坏图 / 空文件

---

## 📂 数据格式说明（非常重要）

### 1️⃣ label.txt（标签定义）

每一行表示一个标签，**行号即类别索引**：

```text
0 正常
1 卡通-卡通-卡通
2 色情-性行为-SM
```

* 第一列是索引（可选，仅用于可读性）
* 标签名支持中文
* **标签总数 = 行数**

---

### 2️⃣ 训练 / 验证 TXT（多标签）

用于训练或验证，每一行格式为：

```text
/path/to/image.jpg 0,1,0,0,1,0
```

说明：

* 第一列：图片路径（绝对或相对）
* 第二列：多标签 0/1 向量（逗号分隔）
* 向量长度必须 ≥ 标签数（不足会自动补 0）

---

### 3️⃣ 推理 TXT（仅图片路径）

用于**无标签推理**，每一行只有图片路径：

```text
/path/to/image1.jpg
/path/to/image2.jpg
```

---

## ⚙️ 环境依赖

推荐使用 Conda：

```bash
conda env create -n eformer python=3.9 -f environment.yml
conda activate eformer
```

支持 CUDA + 多 GPU（已在 DDP 模式下验证）。

---

## 🚀 多标签训练（TXT）

更具体的内容，请见multi_train.sh，示例（4 卡）：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
  --model efficientformerv2_l \
  --data-set TXT \
  --train-txt /path/to/train.txt \
  --val-txt /path/to/val.txt \
  --label-txt /path/to/label.txt \
  --output_dir efficientformer_multilabel \
  --batch-size 128 \
  --epochs 40
```

### 关键参数说明

| 参数               | 说明           |
| ---------------- | ------------ |
| `--data-set TXT` | 启用 TXT 多标签模式 |
| `--train-txt`    | 训练集 txt      |
| `--val-txt`      | 验证集 txt      |
| `--label-txt`    | 标签定义文件       |
| `--output_dir`   | 模型 & 日志输出目录  |

---

## 📊 验证（有标签）

如果 `val-txt` 含有标签向量，则 `--eval` 会自动计算：

* mAP
* micro-F1 / macro-F1
* 最优阈值（在验证集上搜索）

```bash
--eval
```

---

## 🔮 无标签推理测试

### 功能说明

当测试集 **只有图片路径，没有标签** 时，启用 `predict-only` 模式：

* 模型输出 sigmoid 概率
* 按阈值转成 0/1 向量
* 自动映射 `label.txt` 输出标签名

---

### 测试命令示例
更具体的内容，请见multi_test.sh
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
  --model efficientformerv2_l \
  --resume efficientformer_multilabel/checkpoint_epoch39.pth \
  --eval \
  --data-set TXT \
  --val-txt /path/to/test_only_images.txt \
  --label-txt /path/to/label.txt \
  --output_dir efficientformer_test \
  --predict-only \
  --thr 0.5 \
  --pred-out pred_vec.txt \
  --pred-label-out pred_labels.txt
```

---

### 📄 推理输出文件

#### 1️⃣ pred_vec.txt（向量）

```text
/path/to/img1.jpg 1,0,0
/path/to/img2.jpg 0,1,1
```

#### 2️⃣ pred_labels.txt（标签名）

```text
/path/to/img1.jpg 正常
/path/to/img2.jpg 卡通-卡通-卡通,色情-性行为-SM
```

---

## 🧠 阈值说明（thr）

* 默认 `thr=0.5`
* 判定规则：`sigmoid(logit) >= thr → 1`
* 可根据业务需求调整（如 0.3 / 0.7）

---

## ⚠️ 常见问题

### Q1：为什么会报 `./image_data/train` 不存在？

👉 **必须指定**：

```bash
--data-set TXT
```

否则会走 ImageNet 的 `ImageFolder` 逻辑。

---

### Q2：推理时需要标签吗？

👉 不需要。
只要提供：

* `--val-txt`（图片路径）
* `--label-txt`（用于输出维度 & 标签名）

---

## 📚 引用

如果你在研究或论文中使用本项目，请引用原论文：

```bibtex
@article{li2022efficientformer,
  title={Efficientformer: Vision transformers at mobilenet speed},
  author={Li, Yanyu and others},
  journal={NeurIPS},
  year={2022}
}
```

```bibtex
@inproceedings{li2023rethinking,
  title={Rethinking Vision Transformers for MobileNet Size and Speed},
  author={Li, Yanyu and others},
  booktitle={ICCV},
  year={2023}
}
```
