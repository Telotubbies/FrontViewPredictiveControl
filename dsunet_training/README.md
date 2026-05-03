# DSUNet Training for CARLA Lane Detection

DSUNet (Depthwise Separable UNet) implementation for lane detection on CARLA dataset.

Based on paper: **"End-to-End Deep Learning of Lane Detection and Path Prediction for Real-Time Autonomous Driving"** (arXiv:2102.04738)

## 🎯 Key Features

- **5.16x lighter** than standard UNet
- **1.61x faster** inference
- Depthwise separable convolutions
- Combined loss (Dice + BCE + Focal)
- Data augmentation with Albumentations
- TensorBoard logging
- Automatic train/val/test split (70/15/15)

---

## 📁 Project Structure

```
dsunet_training/
├── model/
│   └── dsunet.py           # DSUNet architecture
├── dataset/
│   └── carla_lane_dataset.py  # Dataset loader
├── utils/
│   ├── losses.py           # Loss functions
│   └── metrics.py          # Evaluation metrics
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── run_training.sh         # Training launcher
├── run_evaluation.sh       # Evaluation launcher
├── requirements.txt        # Dependencies
└── README.md              # This file
```

---

## 📊 Dataset

**Path:** `/home/supawich/Desktop/CARLA_0.9.16/carla_lstm_mpc_project/data/lane_train_diverse_20260227_121538`

**Statistics:**
- Total samples: 19,988
- Train: 13,991 (70%)
- Val: 2,998 (15%)
- Test: 2,999 (15%)

**Format:**
- Images: RGB, 640x480
- Masks: Binary, 640x480
- States: CSV with vehicle states

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /home/supawich/Desktop/CARLA_0.9.16/carla_lstm_mpc_project/dsunet_training
source ../../venv/bin/activate
pip install -r requirements.txt
```

### 2. Train Model

```bash
chmod +x run_training.sh
./run_training.sh
```

Or manually:

```bash
python train.py \
    --data-root /path/to/dataset \
    --output-dir ./outputs/dsunet_carla \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.001 \
    --base-channels 64
```

### 3. Monitor Training

```bash
tensorboard --logdir ./outputs/dsunet_carla_XXXXXX/logs
```

### 4. Evaluate Model

```bash
# Update checkpoint path in run_evaluation.sh first
./run_evaluation.sh
```

Or manually:

```bash
python evaluate.py \
    --data-root /path/to/dataset \
    --checkpoint ./outputs/best_model_iou.pth \
    --output-dir ./eval_results \
    --save-vis
```

---

## 📈 Training Details

### Model Architecture

```python
DSUNet(
    in_channels=3,
    num_classes=1,
    base_channels=64
)
# Parameters: ~6M (vs ~31M for standard UNet)
```

### Loss Function

```python
CombinedLoss(
    dice_weight=0.5,   # Dice loss
    bce_weight=0.3,    # Binary cross-entropy
    focal_weight=0.2   # Focal loss (handles imbalance)
)
```

### Optimizer

```python
Adam(
    lr=1e-3,
    weight_decay=1e-4
)

ReduceLROnPlateau(
    factor=0.5,
    patience=5
)
```

### Data Augmentation

- Horizontal flip (p=0.5)
- Random brightness/contrast (p=0.3)
- Gaussian noise (p=0.2)
- Blur (p=0.2)
- Normalization (ImageNet stats)

---

## 📊 Evaluation Metrics

- **IoU (Intersection over Union)**
- **Dice Coefficient**
- **Precision**
- **Recall**
- **F1 Score**

---

## 🎯 Expected Results

Based on paper and similar implementations:

| Metric | Expected | Target |
|--------|----------|--------|
| IoU | 0.75-0.85 | > 0.80 |
| Dice | 0.85-0.90 | > 0.85 |
| F1 | 0.85-0.90 | > 0.85 |

---

## 📝 Training Logs

Training logs are saved to:
- TensorBoard: `outputs/dsunet_carla_XXXXXX/logs/`
- Checkpoints: `outputs/dsunet_carla_XXXXXX/*.pth`

Saved models:
- `best_model_loss.pth` - Best validation loss
- `best_model_iou.pth` - Best validation IoU
- `checkpoint_epoch_N.pth` - Periodic checkpoints

---

## 🔧 Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python train.py --batch-size 8  # or 4
```

### Slow Training

Reduce number of workers:
```bash
python train.py --num-workers 2
```

### Poor Performance

Try:
1. Train longer (--epochs 150)
2. Adjust learning rate (--lr 0.0005)
3. Increase base channels (--base-channels 128)

---

## 📚 References

1. **DSUNet Paper:** https://arxiv.org/abs/2102.04738
2. **Depthwise Separable Convolutions:** Xception paper
3. **Loss Functions:** Dice, Focal Loss papers

---

## 🎓 Next Steps

After training:

1. **Evaluate on test set**
   ```bash
   ./run_evaluation.sh
   ```

2. **Export to ONNX** (for deployment)
   ```python
   torch.onnx.export(model, dummy_input, "dsunet.onnx")
   ```

3. **Integrate with CARLA MPC**
   ```python
   # Replace UNet in lane_detector.py
   from dsunet_training.model.dsunet import DSUNet
   model = DSUNet()
   model.load_state_dict(torch.load("best_model_iou.pth"))
   ```

4. **Fine-tune if needed**
   - Adjust loss weights
   - Try different augmentations
   - Experiment with learning rate schedule

---

## ✅ Checklist

- [x] Dataset prepared (19,988 samples)
- [x] Model architecture implemented
- [x] Loss functions defined
- [x] Training script ready
- [x] Evaluation script ready
- [ ] **Run training** ← START HERE
- [ ] Monitor training progress
- [ ] Evaluate on test set
- [ ] Integrate with CARLA MPC

---

**Ready to train! Run `./run_training.sh` to start.** 🚀
