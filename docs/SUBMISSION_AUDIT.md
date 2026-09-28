# Public submission audit

Prepared on 2026-09-28 from scientific audit milestone `0494ddb5ef13ec7c766c0971b455dcbae003873e`. The development tree already contained untracked report assets; these were not reset or edited. Its Git status also reported permission warnings for old temporary test-output directories. Those directories were excluded from this copy.

## Checks performed

- Every copied source, test, configuration and aggregate file was compared byte-for-byte by SHA256 with its original.
- All included Python files were parsed without importing or executing project code.
- The entire copy's UTF-8 text was scanned for irrelevant tool/chat residue, personal paths, email addresses, credential patterns, identifiers and DICOM identity fields. No irrelevant residue, personal project path, email address or credential was found by these checks. Pattern scanning is not a guarantee that all possible secrets are detectable.
- Four public CMMD pseudonymous constants remain in `src/external_validation/cmmd_cohort.py` as the exact frozen known-duplicate exclusion rule. Removing these would change cohort behaviour. Matching tests exercise this rule. Other identifier-like literals in tests are synthetic fixtures, not copied clinical records.
- DICOM identity field names occur in validation/privacy tests and protective audit code; they do not expose real names, dates of birth or accession values. Deliberately invalid absolute paths in traversal-rejection tests remain unchanged.
- The one generic output-generation phrase in `src/reporting/localisation_final.py:145` describes overlays produced by the scientific evaluator. It is legitimate scientific provenance, not a development-tool attribution. Substring matches inside `fullmatch` are Python regular-expression API names.
- No raw DICOM, clinical workbook, ROI/image cache, model checkpoint, archive, environment, bytecode or Git history is present. The four figures are aggregate SVG charts, with no embedded medical-image panels.
- No included file exceeds 25 MiB, 50 MiB or 95 MiB. The excluded checkpoint is 214,670,418 bytes; see `MODEL_AND_RUNTIME.md`.
- The observed installed dependencies pass `pip check`; no fresh installation or full suite was run for this copy.

## Preserved scope and publication limits

All source/tests remain unchanged. Existing internal READMEs, draft reports, working notes, per-case evidence and history were excluded through an allowlist. No required disclosure identified in the reviewed project files was removed; university requirements still need confirmation. Scientific provenance qualifications are retained in the new public README and result notes.

The README, setup specifications, data citations, dependency/environment notes, manual-upload instructions and private runtime inventory are complete. Checksum verification covers the distribution only. It does not replace the scientific archive verifier or prove a new clean-environment test pass.

Status: prepared for manual review and source/aggregate-evidence upload. Full standalone app/data reproducibility is conditional on authorised runtime files and datasets. Software/weight licensing, institutional disclosure confirmation, fresh-install testing and manual UI acceptance remain open. No publication, new Git repository, scientific execution or original-project edit occurred.
