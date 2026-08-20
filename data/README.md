# Dataset

This project uses the **RSNA Pneumonia Detection Challenge** dataset.

Not committed to this repository due to size. To obtain it:

1. Kaggle competition page: search "RSNA Pneumonia Detection Challenge" on
   kaggle.com, or use the Kaggle API:
   ```bash
   kaggle competitions download -c rsna-pneumonia-detection-challenge
   ```
2. Extract into this `data/` folder so the notebook's `DATA_DIR` config
   points at:
   - `data/stage_2_train_labels.csv`
   - `data/stage_2_detailed_class_info.csv`
   - `data/stage_2_train_images/` (folder of `.dcm` files)

A Kaggle account and acceptance of the competition rules is required to
download the data.
