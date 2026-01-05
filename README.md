# EfficientFormer Multi-Label Classification

This project is based on the **official implementation of EfficientFormer / EfficientFormerV2**, and is **engineered specifically for multi-label image classification** scenarios. It supports:

* 📄 Dataset definition via **TXT files** (no dependency on ImageFolder directory structure)
* 🔢 **Multi-label 0/1 vector training**
* 🔮 **Inference with image paths only (no labels required)**
* 📝 Automatic inference outputs:

  * `image_path + 0,1,0,...`
  * `image_path + label_name1,label_name2,...`

It is suitable for **content moderation, multi-attribute recognition, and industrial multi-label classification** tasks.

In practical evaluations, the **Efficient-l** model trained with this framework achieved a **5%** improvement in mAP across all labels on a pornographic image multi-label classification dataset.
For the binary classification task (normal vs. porn), the model reached over **90%** accuracy and **80%** recall.
---

## 📌 Model Origin

The models used in this project are derived from the following papers:

* **EfficientFormer** – *Vision Transformers at MobileNet Speed* (NeurIPS 2022)
* **EfficientFormerV2** – *Rethinking Vision Transformers for MobileNet Size and Speed* (ICCV 2023)

The official repositories provide highly efficient Vision Transformer architectures.
This project **does NOT modify the model architecture**, and only adapts:

* Dataset loading
* Loss function
* Inference pipeline

Please refer to the official README for the original project description.
https://github.com/snap-research/EfficientFormer
---

## 🧩 Project Features

* ✅ Multi-label classification with `BCEWithLogitsLoss`
* ✅ Arbitrary number of labels (automatically inferred from `label.txt`)
* ✅ Flexible TXT-based dataset format
* ✅ Distributed training / inference (DDP)
* ✅ Label-free inference
* ✅ Automatic skipping of corrupted or empty images

---

## 📂 Dataset Format (Very Important)

### 1️⃣ `label.txt` (Label Definition)

Each line represents **one label**, and **the line index corresponds to the class index**:

```text
0 normal
1 cartoon-cartoon-cartoon
2 pornography-sexual_behavior-SM
```

Notes:

* The first column (index) is optional and for readability only
* Label names support non-English languages (e.g., Chinese)
* **Number of labels = number of lines**

---

### 2️⃣ Training / Validation TXT (Multi-Label)

Each line represents one image and its multi-label annotation:

```text
/path/to/image.jpg 0,1,0,0,1,0
```

Explanation:

* First field: image path (absolute or relative)
* Second field: multi-label 0/1 vector (comma-separated)
* If the vector length is **shorter than the number of labels**, it will be automatically padded with zeros

---

### 3️⃣ Inference TXT (Image Paths Only)

For **label-free inference**, each line contains only an image path:

```text
/path/to/image1.jpg
/path/to/image2.jpg
```

---

## ⚙️ Environment Setup

Conda is recommended:

```bash
conda env create -n eformer python=3.9 -f environment.yml
conda activate eformer
```

* CUDA supported
* Multi-GPU & DDP verified

---

## 🚀 Multi-Label Training (TXT)

For more details, see `multi_train.sh`.

Example (4 GPUs):

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

### Key Arguments

| Argument         | Description                        |
| ---------------- | ---------------------------------- |
| `--data-set TXT` | Enable TXT-based multi-label mode  |
| `--train-txt`    | Training dataset TXT file          |
| `--val-txt`      | Validation dataset TXT file        |
| `--label-txt`    | Label definition file              |
| `--output_dir`   | Output directory for logs & models |

---

## 📊 Validation (With Labels)

If `val-txt` contains label vectors, enabling `--eval` will automatically compute:

* mAP
* Micro-F1 / Macro-F1
* Optimal threshold (searched on validation set)

```bash
--eval
```

---

## 🔮 Label-Free Inference

### Functionality

When the test set **contains only image paths**, enable `predict-only` mode:

* Model outputs sigmoid probabilities
* Probabilities are converted to 0/1 using a threshold
* Label names are automatically mapped using `label.txt`

---

### Inference Command Example

For more details, see `multi_test.sh`.

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

## 📄 Inference Output Files

### 1️⃣ `pred_vec.txt` (Binary Vectors)

```text
/path/to/img1.jpg 1,0,0
/path/to/img2.jpg 0,1,1
```

---

### 2️⃣ `pred_labels.txt` (Label Names)

```text
/path/to/img1.jpg normal
/path/to/img2.jpg cartoon-cartoon-cartoon,pornography-sexual_behavior-SM
```

---

## 🧠 Threshold (`thr`) Explanation

* Default: `thr = 0.5`

* Decision rule:

  ```
  sigmoid(logit) >= thr → 1
  ```

* Can be adjusted based on business requirements (e.g., 0.3 / 0.7)

---

## ⚠️ Common Issues

### Q1: Why does it complain that `./image_data/train` does not exist?

👉 You **must specify**:

```bash
--data-set TXT
```

Otherwise, the default ImageNet `ImageFolder` logic will be used.

---

### Q2: Are labels required during inference?

👉 **No.**

You only need:

* `--val-txt` (image paths)
* `--label-txt` (for output dimension & label name mapping)

---

## 📚 Citation

If you use this project in research or publications, please cite the original papers:

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