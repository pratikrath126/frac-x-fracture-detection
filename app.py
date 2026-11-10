from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.explain import fracture_gradcam, overlay_gradcam
from src.model import load_checkpoint

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "artifacts" / "best_model.pt"

st.set_page_config(page_title="FRAC-X | Fracture X-ray Demo", page_icon="🦴", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1240px; padding-top: 1.8rem; padding-bottom: 3rem;}
.hero {padding: 1.35rem 1.6rem; border-radius: 18px; background: linear-gradient(115deg,#10233f,#17627b); color: white; margin-bottom: 1rem;}
.hero h1 {font-size: 2.35rem; margin: 0 0 .25rem 0; color: white;}
.hero p {font-size: 1.02rem; opacity: .92; margin: 0;}
.warning {border-left: 4px solid #e9a800; background: #fff7e4; padding: .8rem 1rem; border-radius: 8px;}
[data-testid="stMetric"] {background: #f5f8fc; border: 1px solid #e6ebf2; padding: 13px 16px; border-radius: 12px;}
</style>
<div class="hero"><h1>FRAC-X</h1><p>Bone fracture X-ray classification · pooled FracAtlas + BoneFract model</p></div>
<div class="warning"><b>Academic prototype only.</b> This model is not a medical diagnostic tool. Do not use predictions to guide care.</div>
""", unsafe_allow_html=True)


def read_json(relative_path: str, fallback):
    try:
        return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


@st.cache_resource
def get_model(path: str, device_name: str):
    return load_checkpoint(path, torch.device(device_name))


@st.cache_data
def get_reports():
    return (
        read_json("outputs/test_metrics.json", {}),
        read_json("outputs/training_history.json", []),
        read_json("outputs/validation_metrics.json", {}),
        read_json("outputs/training_metrics.json", {}),
    )


test, history, valid, training = get_reports()
trials = read_json("outputs/local_model_trials.json", {})
if not CHECKPOINT.is_file():
    st.error("The trained checkpoint is missing: artifacts/best_model.pt")
    st.stop()

live_tab, report_tab, info_tab = st.tabs(["Live prediction", "Performance dashboard", "Model & data"])

with live_tab:
    st.header("Upload an X-ray")
    st.write("Upload a JPG, JPEG, or PNG. The model resizes it to 224 × 224 RGB and applies the same ImageNet normalization used during training.")
    input_col, output_col = st.columns([1, 1], gap="large")
    with input_col:
        upload = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])
        default_threshold = float(test.get("threshold_from_validation", 0.6641))
        threshold = st.slider("Decision threshold", 0.05, 0.95, min(0.95, max(0.05, default_threshold)), 0.01,
                              help="Lower values flag more images as fracture and usually increase false alarms.")
        st.caption(f"Default threshold selected on pooled validation: {default_threshold:.3f}")
    if upload is None:
        with output_col:
            st.info("The prediction and Grad-CAM explanation will appear here after upload.")
    else:
        try:
            image = Image.open(upload).convert("RGB")
            if min(image.size) < 64:
                raise ValueError("Image is too small. Upload an X-ray at least 64 pixels in each dimension.")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, checkpoint = get_model(str(CHECKPOINT), str(device))
            with st.spinner("Analyzing the image and building its Grad-CAM…"):
                cam, score = fracture_gradcam(model, checkpoint["model_name"], image, device)
                overlay = overlay_gradcam(image, cam)
            with input_col:
                st.image(image, caption="Uploaded image", use_container_width=True)
                st.metric("Fracture score", f"{score:.1%}", help="Model score, not a calibrated clinical probability.")
                st.progress(min(max(score, 0.0), 1.0), text=f"Score {score:.1%} · threshold {threshold:.2f}")
                if score >= threshold:
                    st.error("Prediction: fracture pattern")
                else:
                    st.success("Prediction: non-fracture pattern")
            with output_col:
                st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
                st.caption("Grad-CAM highlights regions influencing the model score; it does not confirm or locate a fracture.")
        except (OSError, ValueError) as exc:
            st.error(f"Could not process this upload: {exc}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with report_tab:
    st.header("Performance dashboard")
    st.caption("ResNet-50 trained on the pooled training sets. The selected checkpoint and threshold use pooled validation accuracy.")
    if not test:
        st.warning("Saved pooled test metrics are missing. Run evaluate.py to generate them.")
    else:
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Test accuracy", f"{test.get('accuracy', 0):.1%}")
        k2.metric("Train accuracy*", f"{training.get('accuracy', 0):.1%}" if training else "Unavailable")
        k3.metric("Precision", f"{test.get('precision', 0):.1%}")
        k4.metric("Fracture recall", f"{test.get('fracture_recall_sensitivity', 0):.1%}")
        k5.metric("F1", f"{test.get('f1', 0):.3f}")
        k6.metric("ROC-AUC", f"{test.get('roc_auc', 0):.3f}")
        st.caption("*Training-pool accuracy calculated after training. Per-epoch training accuracy and total training time were not recorded for this run.")
        st.caption(f"Pooled test: {test.get('sample_count', 0):,} images · specificity {test.get('specificity', 0):.1%} · PR-AUC {test.get('pr_auc', 0):.3f} · decision threshold {test.get('threshold_from_validation', 0):.3f}")

        cm_col, roc_col = st.columns(2, gap="large")
        cm = np.asarray(test.get("confusion_matrix_labels_0_1", [[0, 0], [0, 0]]), dtype=int)
        with cm_col:
            st.subheader("Confusion matrix")
            fig, ax = plt.subplots(figsize=(5.6, 4.2))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1], labels=["Non-fracture", "Fracture"])
            ax.set_yticks([0, 1], labels=["Non-fracture", "Fracture"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Pooled held-out test")
            max_value = max(1, int(cm.max()))
            for (row, col), value in np.ndenumerate(cm):
                ax.text(col, row, f"{value:,}", ha="center", va="center",
                        color="white" if value > max_value * .52 else "#17324d", fontsize=13, fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            tn, fp, fn, tp = cm.ravel()
            st.write(f"True negatives **{tn:,}** · False positives **{fp:,}** · False negatives **{fn:,}** · True positives **{tp:,}**")
        with roc_col:
            st.subheader("ROC curve")
            roc = test.get("roc_curve", {})
            if roc.get("fpr") and roc.get("tpr"):
                fig, ax = plt.subplots(figsize=(5.6, 4.2))
                ax.plot(roc["fpr"], roc["tpr"], lw=2, color="#168aad", label=f"Pooled test (AUC {test['roc_auc']:.3f})")
                ax.plot([0, 1], [0, 1], "--", color="#888", label="Chance")
                ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate", title="Pooled held-out test")
                ax.grid(alpha=.2)
                ax.legend(loc="lower right")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("ROC-AUC is recorded; detailed ROC curve coordinates are not available in this evaluation file.")

        st.subheader("Training and validation curves")
        if history:
            epochs = [row.get("epoch", idx + 1) for idx, row in enumerate(history)]
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
            axes[0].plot(epochs, [row.get("train_loss") for row in history], marker="o", label="Train loss")
            axes[0].plot(epochs, [row.get("validation_loss") for row in history], marker="o", label="Validation loss")
            axes[0].set(title="Loss by epoch", xlabel="Epoch", ylabel="Binary cross-entropy")
            axes[0].legend()
            axes[0].grid(alpha=.2)
            axes[1].plot(epochs, [row.get("validation_roc_auc") for row in history], marker="o", label="Validation ROC-AUC")
            axes[1].plot(epochs, [row.get("validation_pr_auc") for row in history], marker="o", label="Validation PR-AUC")
            axes[1].set(title="Validation ranking metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
            axes[1].legend()
            axes[1].grid(alpha=.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        st.info("Training-pool accuracy is calculated after training and is reported separately from held-out validation and test metrics. Epoch-wise training accuracy and elapsed training time were not recorded for this run.")

        st.subheader("Main architecture comparison")
        if trials.get("candidates"):
            st.info(
                "Three pretrained architectures compared on the pooled validation set. "
                "Each threshold was chosen separately to target 80% fracture recall; the live ResNet-50 "
                "predictor above keeps its accuracy-selected threshold."
            )
            trial_rows = []
            main_architectures = ["ResNet-50", "ConvNeXt-Tiny", "EfficientNet-B0"]
            candidates_by_name = {candidate["name"]: candidate for candidate in trials["candidates"]}
            for name in main_architectures:
                candidate = candidates_by_name.get(name)
                if not candidate:
                    continue
                point = candidate["recall_targets"]["80"]
                trial_rows.append({
                    "Architecture": name,
                    "Validation accuracy": f"{point['accuracy']:.1%}",
                    "Fracture recall": f"{point['recall']:.1%}",
                    "Precision": f"{point['precision']:.1%}",
                    "ROC-AUC": f"{candidate['validation_roc_auc']:.3f}",
                })
            st.dataframe(
                trial_rows,
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                f"Validation set: {trials.get('validation_count', 0):,} pooled images. "
                "These validation results were used during model selection, not as an independent final estimate."
            )
        else:
            st.info("Architecture comparison results are not available in this deployment.")

        st.subheader("Pooled test breakdown")
        source_rows = []
        for name, values in test.get("per_dataset", {}).items():
            source_rows.append({"Source": name, "Images": values.get("sample_count"), "Accuracy": values.get("accuracy"),
                                "Fracture recall": values.get("fracture_recall_sensitivity"), "Specificity": values.get("specificity"),
                                "ROC-AUC": values.get("roc_auc")})
        if source_rows:
            st.dataframe(source_rows, hide_index=True, use_container_width=True,
                         column_config={key: st.column_config.NumberColumn(format="%.1%") for key in ["Accuracy", "Fracture recall", "Specificity"]})
        st.caption("The pooled score follows the test-set mix. FracAtlas test images were viewed during earlier development; BoneFract test images remained held out. Results do not guarantee performance on new hospitals or patient groups.")

with info_tab:
    st.header("Model and data")
    st.markdown("### Dataset used")
    total_col, train_col, val_col, test_col = st.columns(4)
    total_col.metric("Total pooled images", "53,846")
    train_col.metric("Training pool", "43,038")
    val_col.metric("Validation", "5,403")
    test_col.metric("Held-out test", "5,405")
    st.caption("The confusion matrix and pooled test metrics on the dashboard use the 5,405-image test split.")
    split_rows = [
        {"Dataset": "FracAtlas", "Train": "4,694", "Validation": "610", "Test": "611", "Total": "5,915"},
        {"Dataset": "BoneFract", "Train": "38,344", "Validation": "4,793", "Test": "4,794", "Total": "47,931"},
        {"Dataset": "Combined", "Train": "43,038", "Validation": "5,403", "Test": "5,405", "Total": "53,846"},
    ]
    st.dataframe(split_rows, hide_index=True, use_container_width=True)
    st.markdown("**Training sampling:** the training pool contains 2,347 FracAtlas fracture images (including 1,847 augmented views), 2,347 FracAtlas non-fracture images, 11,192 BoneFract fracture images, and 27,152 BoneFract non-fracture images. Balanced sampling draws 6,000 examples from each of these four groups per epoch (24,000 draws total), with replacement. The full training pool is eligible for sampling, but individual images are not guaranteed to appear in every epoch.")
    st.markdown("### Model setup")
    st.markdown("**Architecture:** ImageNet-pretrained ResNet-50, fine-tuned on the pooled FracAtlas and BoneFract training partitions.\n\n**Preprocessing:** RGB, 224 × 224, ImageNet mean and standard deviation.\n\n**Validation and model selection:** pooled FracAtlas + BoneFract validation partitions; maximum pooled accuracy selects the threshold.\n\n**Final evaluation:** pooled held-out test partitions. The source-specific breakdown is diagnostic; the model itself is a single pooled classifier.")
    st.markdown("**BoneFract attribution:** Noor & Ali, Mendeley Data v1, CC BY 4.0. [DOI 10.17632/4cr7f2359x.1](https://doi.org/10.17632/4cr7f2359x.1). See the project README for FracAtlas and BoneFract provenance.")
    st.markdown("**Metric note:** accuracy, precision, and fracture recall describe different aspects of performance. The default threshold was selected to maximize accuracy on pooled validation data.")
    st.markdown("**Grad-CAM note:** a visual aid, not proof of a fracture or a clinical explanation.")
