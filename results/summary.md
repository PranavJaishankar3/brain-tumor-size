# Test-set results (held-out patients)

| Run | Method | Preproc | Dice | IoU | Area MAE | Size acc (no post-proc) | Size acc | Size macro-F1 | Consistency |
|---|---|---|---|---|---|---|---|---|---|
| seg_raw | segment→measure | raw | 0.720 | 0.631 | 0.012 | 0.792 | 0.794 | 0.795 | 0.951 |
| seg_zscore | segment→measure | zscore | 0.700 | 0.608 | 0.014 | 0.761 | 0.776 | 0.777 | 0.968 |
| seg_clahe | segment→measure | clahe | 0.686 | 0.593 | 0.014 | 0.749 | 0.757 | 0.759 | 0.978 |
| cls_clahe | direct CNN | clahe | – | – | – | – | 0.724 | 0.729 | 0.979 |
| seg_raw_big | segment→measure | raw | 0.762 | 0.679 | 0.011 | 0.790 | 0.786 | 0.784 | 0.971 |

Consistency = share of test scans whose predicted size category is unchanged under 4 simulated scanner intensity shifts (gamma/contrast).
