# FRAC-X | Fracture X-ray Demo

A Streamlit classroom demo for a single pooled fracture classifier. The selected ImageNet-pretrained ResNet-50 uses pooled FracAtlas and BoneFract training data. Model selection and the default decision threshold are based on pooled validation accuracy.

## Run locally

Use Python 3.11 or newer. Install the dependencies and launch the app:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app loads `artifacts/best_model.pt`, which is included. It supports JPG/JPEG/PNG uploads, a live threshold slider, the fracture score and class prediction, and a Grad-CAM overlay. The performance dashboard includes pooled test metrics, confusion matrix, ROC curve, training/validation loss curves, candidate comparison, and a per-source diagnostic breakdown.

## Deploy on Streamlit Community Cloud

1. Put the contents of this `Project` folder at the root of a GitHub repository. Keep `app.py`, `requirements.txt`, `.streamlit/config.toml`, `src/`, `artifacts/best_model.pt`, and the report JSON files together.
2. In Streamlit Community Cloud, create an app for that repository and select `app.py` as the entrypoint. Select Python 3.11 in the deployment's advanced settings.
3. The image dataset is intentionally kept out of the app repository. Live prediction uses the included checkpoint and does not need the training/test images.

## Reproduce training and evaluation locally

The full dataset is in the adjacent `../Dataset/` folder, split into `Fracture/` and `Non-Fracture/` (53,846 images). CSV manifests in `outputs/` map each image to its official source and train/validation/test split. Keep `Dataset` beside `Project` when training or reevaluating.

| Data source | Training pool | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| FracAtlas | 4,694 | 610 | 611 | 5,915 |
| BoneFract | 38,344 | 4,793 | 4,794 | 47,931 |
| **Combined** | **43,038** | **5,403** | **5,405** | **53,846** |

The model was fitted using the 43,038-image training pool. Balanced sampling draws 6,000 examples from each of the four dataset/class groups per epoch (24,000 draws total), with replacement, so some images may repeat and some may not be drawn during a given epoch. The 5,403-image validation split was used for model/threshold selection; the 5,405-image test split is what the confusion matrix and pooled test metrics summarize.

```powershell
python train.py --model resnet50 --epochs 12 --workers 6 --include-bonefract --output-dir artifacts/resnet50_combined
python measure_training_metrics.py
python evaluate.py --include-bonefract --checkpoint artifacts/best_model.pt --output outputs/test_metrics.json
```

Training balances four source/class groups during sampling. Validation and test partitions remain separate from training. The final app checkpoint was selected on pooled validation. The FracAtlas test portion was viewed during earlier development, so it is not a pristine independent benchmark; BoneFract test remained held out.

## Current pooled test results

On 5,405 pooled test images, the selected checkpoint recorded **85.3% accuracy**, **83.2% precision**, **59.0% fracture recall**, **0.691 F1**, and **0.874 ROC-AUC** at threshold 0.664. The test set contains both datasets, with BoneFract contributing most images. The full pooled and per-source metrics are in `outputs/test_metrics.json`.

The completed run logged train/validation loss, validation ROC-AUC/PR-AUC, and epoch count. A post-training calculation recorded **86.8% accuracy on the training pool** at the selected threshold. This value is shown separately from held-out metrics. Epoch-wise training accuracy and elapsed training time were not recorded for this run.

## Project scope from the FRAC-X plan

The demo follows the plan's transfer-learning, image-upload, confidence-score, Grad-CAM, test metrics, confusion matrix, ROC, and false-positive/false-negative reporting goals. A pretrained ResNet-50 was selected for the simple integrated project; no scratch baseline CNN or separate attention model is included. The app is an academic prototype, not a medical diagnostic tool.

## Dataset attribution

BoneFract: Noor, R. & Ali, A. (2026), *BoneFract: A Bone Fracture Dataset*, Mendeley Data v1, CC BY 4.0, [DOI 10.17632/4cr7f2359x.1](https://doi.org/10.17632/4cr7f2359x.1). Review each dataset's terms before redistributing its images. Dataset images are not included in this repository package.
