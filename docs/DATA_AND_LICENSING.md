# Data access, citations and redistribution

Checked against official documentation on 2026-09-28. This copy contains no raw CBIS-DDSM/CMMD images, cropped lesions, ROI masks, clinical XLSX, participant-level predictions or medical-image exports. Data and model access are separate from source-code access.

- [CBIS-DDSM, TCIA](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) lists the collection and clinical CSVs under CC BY 3.0. Cite Sawyer-Lee, Gimenez, Hoogi and Rubin (2016), *Curated Breast Imaging Subset of Digital Database for Screening Mammography*, DOI [10.7937/K9/TCIA.2016.7O02S9CY](https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY), and the collection's requested publications.
- [CMMD, TCIA](https://www.cancerimagingarchive.net/collection/cmmd/) lists the imaging collection under CC BY 4.0. Obtain the official clinical workbook and imaging through its access links and use the collection's current required data/publication citations.
- Follow [TCIA usage policies](https://wiki.cancerimagingarchive.net/display/Public/Data+Usage+Policies+and+Restrictions), including attribution and restrictions on identification/contact of participants. Confirm current collection-specific terms when obtaining data.

For independent research, obtain CBIS-DDSM official Mass-Training and Mass-Test clinical metadata and full-mammogram series; ROI resources are only for alignment/localisation assessment. CMMD uses the official revised clinical spreadsheet plus series matched to a frozen side-specific MASS cohort. Do not infer contralateral labels. Obtaining a complete collection is not equivalent to reconstructing this project's locked cohort: the authorised original manifests and hashes are also required.

No raw data are downloaded by these instructions. Use `data/raw/CBIS-DDSM/` and `data/external/CMMD/` locally; keep them out of public uploads. `data/README.md` describes scope.

The project code has no verified project-wide licence grant in the source repository; the author/institution must choose one before claiming an open-source licence. Do not apply a blanket licence to dataset material or external dependencies. [Keras Applications](https://keras.io/api/applications/) supplies pretrained architectures/weights, but a software licence does not by itself settle every ImageNet/pretrained/fine-tuned weight redistribution condition. Weight redistribution clearance remains unresolved; no model file or licence grant has been fabricated here.

Academic disclosure requirements were not supplied or independently verified. Confirm them with the module/supervisor before publication; retain any required declarations. Omitting internal working notes does not establish independent authorship or waive required disclosures.
