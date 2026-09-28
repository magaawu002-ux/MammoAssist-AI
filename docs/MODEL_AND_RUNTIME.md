# Model installation and runtime evidence

The selected ResNet50 checkpoint is **214,670,418 bytes** (204.726 MiB), SHA256 `0076686617cb29ad5e5682f3d1eff89fae9cad2cebffa86e2b8be2b537934384`. It is excluded from this copy. The exact local target is `results/experiments/resnet50/20260925T083032_895231Z/best_model.keras`. Keep threshold `0.3753475248813629`, canonical_v1 preprocessing and the original source unchanged.

Recommendation: keep the checkpoint local until redistribution is cleared, then use a checksum-verified GitHub Release asset or approved external download. Do not add it through the repository browser upload: its size exceeds both the 25 MiB browser limit and the 100 MiB ordinary Git limit. Git LFS requires a separate client workflow and is not supplied by a browser upload. A Release can technically host this file, but no permission or public download URL is implied.

The app's `src/app/core.py:contract` also requires the following original files. **Copy the original bytes privately; do not redact or fabricate replacement locks.**

- `results/experiments/final_selection.json`
- `results/explainability/localisation_lock.json`
- `reports/evidence/roi_alignment_protocol.json`
- `reports/evidence/preprocessing_validation.json`
- `reports/evidence/final_localisation/summary.json`
- `reports/evidence/robustness/stability_summary.json`

The first four files bind selection, localisation and preprocessing. The last two support the frozen dashboard. Some contain pseudonymous participants, source inventories or per-case details; the set is excluded as a restricted runtime bundle rather than rewritten. Exact sizes/hashes are in [private_runtime_requirements.json](private_runtime_requirements.json). Further private data are needed by full scientific verifiers, beyond this app bundle.

Approved private recipients can obtain these files from the project maintainer and restore them at the exact relative paths. This public copy alone will fail the app contract, by design. Do not weaken the check merely to make a screenshot. Keep restored files out of any future manual browser upload; `.gitignore` does not filter browser selections.

Copied aggregate dashboard CSVs and sanity summaries retain their original paths. Source and configuration are unmodified. Full data pipelines, locked final evaluation and the full-archive verifier need additional omitted data/evidence; they are not stand-alone reproduction commands for this subset.
