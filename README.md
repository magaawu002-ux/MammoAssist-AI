# MammoAssist AI

MammoAssist AI is a final-year Computer Science research project investigating full-mammogram benign/malignant mass classification and Grad-CAM explanation. It follows **CM3015 Machine Learning and Neural Networks ? Project Idea 2** and supports the CM3070 final project report.

The research question is whether a model developed on CBIS-DDSM mass cases can retain useful discrimination on an independent CMMD cohort, and how reliably its attribution maps align with lesion annotations.

**Research prototype only: not a medical device or an autonomous diagnostic system.** Scores are not calibrated clinical cancer probabilities. Grad-CAM visualises model attribution; it does not detect lesions or establish tumour boundaries.

## Features and architecture

- Participant-safe CBIS-DDSM development splits and full-mammogram preprocessing.
- Majority reference, two custom CNNs and VGG16/ResNet50/EfficientNetB0 comparisons.
- Frozen ResNet50 with the research threshold **0.3753475248813629**.
- Genuine Grad-CAM at `conv5_block3_out`, with a malignant sigmoid-score target.
- Separate localisation, robustness and external-validation implementations.
- Streamlit input review, technical quality heuristics, confirmation, explanations and PNG/PDF export.
- Session-only history and two-case comparison of completed results; a dashboard reads saved aggregate evidence.

Offline research produces locked artefacts. The interactive app uses those artefacts and never fits a model. Source validation, canonical 224 ? 224 grayscale preprocessing and the ResNet50 adapter precede inference. ROI/mask resources are used only for localisation assessment.

## Data and evidence

CBIS-DDSM mass cases supplied 981 Training, 250 Validation and 361 official Test full mammograms. Independent CMMD evaluation included 2,282 images from 1,141 eligible breast sides and 1,090 participants. Clinical labels remain side-specific; 820 contralateral images were excluded.

**Datasets are not redistributed.** This copy excludes raw images, clinical workbooks, participant-level predictions/manifests, ROI resources and caches. Obtain source data independently through TCIA: [CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) and [CMMD](https://www.cancerimagingarchive.net/collection/cmmd/). See [data access and licensing](docs/DATA_AND_LICENSING.md).

## Key findings

| Frozen image-level evaluation | Images | Sensitivity | Specificity | Balanced accuracy | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| CBIS-DDSM official Test | 361 | 0.7172 | 0.4815 | 0.5994 | 0.6711 |
| CMMD external validation | 2,282 | 0.8007 | 0.3090 | 0.5548 | 0.5835 |

CMMD sensitivity increased while specificity, balanced accuracy and discrimination declined. This mixed transfer pattern exposes substantial generalisation limitations; it does not establish clinical utility. Confidence intervals and other metrics are in the [aggregate tables](reports/aggregate/) and [result qualifications](docs/RESULTS_AND_LIMITATIONS.md). No external training, fine-tuning, recalibration or threshold selection was performed.

## Setup

The recorded local environment uses Python 3.12.10 on Windows. From this repository directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
# Only if parsing CMMD workbooks:
.\.venv\Scripts\python.exe -m pip install -r requirements-metadata.txt
```

These pins reflect the observed environment; clean-install and cross-platform validation remain outstanding. Historical GPU training environments are documented separately in [ENVIRONMENT.md](docs/ENVIRONMENT.md).

## Tests and static verification

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q
.\.venv\Scripts\python.exe -B tools/verify_submission.py
```

The earlier full development audit recorded **537 passed, 333 warnings** and a 145-file reproducibility-verifier PASS. These are historical outcomes, not a new test run of this public subset. The static submission verifier checks included-file hashes only; it does not validate scientific results. The original full-archive verifier requires omitted private evidence. Tests include synthetic model exercises, whereas tests/assessment of saved results are not permission to repeat locked experiments.

## Run the Streamlit prototype

First restore the approved checkpoint **and** the unmodified private provenance bundle described in [MODEL_AND_RUNTIME.md](docs/MODEL_AND_RUNTIME.md). This public source/evidence copy alone is not a complete runtime installation: its integrity checks intentionally refuse operation when required files are absent. There is currently no verified public model download URL.

Expected checkpoint:

```text
results/experiments/resnet50/20260925T083032_895231Z/best_model.keras
```

Then launch locally:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/app/streamlit_app.py
```

The supplied configuration binds the server to localhost. It is not production authentication or security certification. Uploads are processed in session memory; downloaded exports can contain medical pixels and must be reviewed for burned-in identifiers before sharing. Manual browser acceptance has not been certified.

## Repository structure

```text
config/        experiment, preprocessing and integrity settings
src/           data, models, research pipelines and Streamlit implementation
tests/         preserved automated tests
docs/          setup, provenance, limitations and data-access notes
reports/       selected aggregate evidence and configuration/history records
results/       selected frozen run metadata and aggregate metrics; no weights
figures/       aggregate report charts; no medical-image panels
tools/         static submission checksum verification
data/, models/ empty required local directory placeholders
```

## Reproducibility and limitations

Scientific source, tests, configuration and copied evidence retain their original bytes. The selected checkpoint hash, private runtime dependencies and copy inventory are documented. This is a curated public subset, not the full scientific archive; full reproducibility also requires authorised data access and the original bound artefacts.

The original global CBIS selection lock and final-Test receipt were reconstructed from preserved artefacts after a reset. Their original pre-Test timestamps cannot be independently verified. The CMMD side-level mean-score analysis is **post-hoc exploratory**, not prespecified. Neither qualification is repaired by rerunning experiments.

Measured localisation overlap is low and the Validation occlusion test did not support the hypothesised faithfulness behaviour. CMMD has a marked unresolved CC/MLO failure difference. No radiologist user study or clinical deployment validation is claimed.

## Licence and citation

No project-wide software licence has yet been selected; public availability alone does not grant an open-source licence. Dataset licences are separate from software and model-weight permissions. Review [DATA_AND_LICENSING.md](docs/DATA_AND_LICENSING.md) before distributing weights or reusing data. Preserve all university-required acknowledgements and disclosures in the academic submission. Manual publication instructions are in [MANUAL_GITHUB_UPLOAD.md](MANUAL_GITHUB_UPLOAD.md).
