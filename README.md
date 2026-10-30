# Brain Tumor Imaging — Size Classification

A computer-vision pipeline that segments the tumor region in brain MRI slices and classifies tumor size
into **small / medium / large** from the visible tumor area. It compares preprocessing methods and
two approaches to size estimation, and measures how consistent each one stays across scans with different intensities.

## Approach

1. **Data**: [Figshare brain tumor dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
   (Cheng et al., 2017). It has 3,064 T1 contrast-enhanced MRI slices from 233 patients, with expert tumor masks.
   The tumor types are meningioma, glioma and pituitary. No Kaggle account is needed.
2. **Head-region mask**: Otsu thresholding, morphology, the largest component and hole filling.
   It is used for intensity normalization and as the reference area.
3. **Size label**: tumor area divided by head area. The categories are tertiles computed on the **training patients only**.
4. **Preprocessing variants** (`--preproc`):
   - `raw`: min-max scaling
   - `zscore`: intensity standardization inside the head region
   - `clahe`: zscore plus contrast-limited adaptive histogram equalization
5. **Segment, then measure** (main method): a U-Net separates the tumor from the surrounding tissue (Dice + BCE loss).
   Post-processing keeps the largest component inside the head. The predicted area is then mapped to a size category.
6. **Baseline**: a ResNet-18 predicts the size category directly from the image, with no segmentation.
7. **Evaluation on held-out patients**: the split is patient-level, so there is no leakage.
   - Dice and IoU of the tumor region
   - Size accuracy and macro-F1
   - Area-fraction MAE
   - Effect of post-processing
   - **Consistency**: the percentage of scans whose size category is unchanged under 4 simulated scanner intensity shifts

## Results (held-out test set: 36 patients, 486 scans)

Trained on an Apple M-series MacBook Air (MPS).

| Model | Preprocessing | Dice | IoU | Area MAE | Size accuracy | Size macro-F1 | Consistency* |
|---|---|---|---|---|---|---|---|
| U-Net (base 32, 40 epochs) → measure | raw | **0.762** | **0.679** | **0.011** | 0.786 | 0.784 | 0.971 |
| U-Net (base 16, 10 epochs) → measure | raw | 0.720 | 0.631 | 0.012 | **0.794** | **0.795** | 0.951 |
| U-Net (base 16, 10 epochs) → measure | z-score | 0.700 | 0.608 | 0.014 | 0.776 | 0.777 | 0.968 |
| U-Net (base 16, 10 epochs) → measure | CLAHE | 0.686 | 0.593 | 0.014 | 0.757 | 0.759 | 0.978 |
| ResNet-18 direct classifier (8 epochs) | CLAHE | – | – | – | 0.724 | 0.729 | **0.979** |

\*Share of test scans whose size category stays the same under 4 simulated scanner intensity shifts (gamma/contrast).

**Takeaways**
- Segmenting the tumor first and then measuring it beats predicting size directly: 79.4% vs. 72.4% accuracy.
- Intensity normalization trades a little accuracy for stability: consistency rises from 95.1% (raw) to 97.8% (CLAHE).
- Keeping only the largest tumor region inside the head improved size accuracy in every run
  (e.g. 0.792 → 0.794 for raw, 0.749 → 0.757 for CLAHE).
- A wider, longer-trained U-Net improved Dice from 0.720 to 0.762.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch from pytorch.org first.
Apple Silicon (MPS) and CPU are detected automatically.

## Run everything

```bash
bash run_all.sh
```

Or step by step:

```bash
python src/download_data.py        # ~880 MB from Figshare
python src/prepare_data.py         # resize to 256, labels, patient split
python src/train_seg.py --preproc clahe --epochs 30
python src/train_seg.py --preproc zscore --epochs 30
python src/train_seg.py --preproc raw --epochs 30
python src/train_cls.py --preproc clahe --epochs 20
python src/evaluate.py runs/seg_raw runs/seg_zscore runs/seg_clahe runs/cls_clahe
```

The results go to `results/summary.md` (comparison table) and `results/*_overlays.png` (ground truth vs. predicted masks).

**Rough time per U-Net epoch** (about 2,150 training images):

| Hardware | Time per epoch |
|---|---|
| Recent NVIDIA GPU | under 1 min |
| Apple M-series | about 1–3 min |
| CPU only | about 10 min |

Short on time? These settings are still meaningful:

```bash
EPOCHS=10 CLS_EPOCHS=8 BASE=16 bash run_all.sh
```

## Predict on a single scan

```bash
python src/predict.py path/to/scan.png --run runs/seg_clahe
```

## Notes and limits

- Size categories are relative (tertiles of this dataset), not clinical thresholds. They are based on 2D slices, not 3D tumor volume.
- The head mask includes the skull, because the dataset is not skull-stripped. It is consistent across scans, which is what the ratio needs.
- This is research and education code, not a medical device.
