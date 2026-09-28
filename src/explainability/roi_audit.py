"""Annotation-only audit and verified alignment, without classifier inference."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pydicom

from src.data.cbis_ddsm_metadata import write_csv
from src.data.provenance import sha256_file
from src.explainability.alignment import prove_full_grid, transform_mask
from src.explainability.coordinate_protocol import PROTOCOL, bound_hash, coordinate_rule, load_coordinate_protocol, locked_cohort
from src.explainability.roi_mapping import annotation_references, discover_annotations, map_annotations, reference_parts
from src.explainability.roi_masks import decode_mask
from src.preprocessing.manifests import local_path, read_rows, verify_full_reference
from src.utils.config import load_config


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def localisation_config(root: Path) -> dict:
    config = load_config(root / "config/localisation.yaml")
    if (config.get("version") != "localisation_v1" or config.get("heatmap_rule") != "fixed_normalised_threshold"
            or config.get("target_layer") != "conv5_block3_out" or not 0 < config.get("heatmap_threshold", 0) <= 1
            or type(config.get("overlay_limit")) is not int or config["overlay_limit"] < 0):
        raise ValueError("Invalid or unsupported prespecified localisation protocol")
    local_path(root, config["roi_root"], ("data", "raw", "CBIS-DDSM", "roi_cropped"))
    local_path(root, config["alignment_reviews"], ("data", "interim"))
    return config


def selected_images(root: Path, partition: str) -> tuple[list[dict], dict]:
    """Load only this partition's manifest; no pixel arrays or classification metrics."""
    manifest = root / f"data/processed/manifests/{partition}_processed.csv"
    rows = read_rows(manifest)
    splits = read_rows(root / f"data/splits/{partition}.csv")
    original = {r["canonical_image_id"]: r for r in splits}
    if len(rows) != len(splits) or len(original) != len(splits) or len({r["canonical_image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate or incomplete image cohort")
    summary = json.loads((root / f"reports/evidence/preprocessing_{partition}.json").read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE" or summary.get("failed") != 0 or summary.get("succeeded") != len(rows):
        raise ValueError("Preprocessing not complete")
    if summary["split_sha256"][partition] != sha256_file(root / f"data/splits/{partition}.csv"):
        raise ValueError("Split changed since preprocessing")
    if rows != read_rows(root / f"reports/evidence/preprocessing_{partition}_attempt.csv"):
        raise ValueError("Processed manifest differs from completed preprocessing log")
    for row in rows:
        source = original.get(row["canonical_image_id"])
        if source is None or any(row.get(k) != v for k, v in source.items()):
            raise ValueError("Processed image traceability changed")
        if (row["final_partition"] != partition or row["official_partition"] != ("test" if partition == "test" else "train")
                or row["processing_status"] != "success" or row["mapping_status"] != "uniquely_mapped"):
            raise ValueError("Invalid image cohort partition/status")
    return rows, {"processed_manifest_sha256": sha256_file(manifest), "split_hashes": summary["split_sha256"],
                  "cache_fingerprint": summary["fingerprint"]}


def audit_roi(root: Path, *, partition: str = "validation", final_authorized: bool = False) -> tuple[dict, list[dict], list[dict]]:
    """Return summary/table/images; missing data never produces invented mapping counts."""
    if partition not in ("train", "validation", "test") or (partition == "test" and not final_authorized):
        raise ValueError("Test annotations require the explicit locked final localisation action")
    coordinate_protocol = load_coordinate_protocol(root)
    config = localisation_config(root)
    if coordinate_protocol and partition == 'test':
        # A coordinate lock is not permission for final localisation, even via this API.
        from src.explainability.localisation import require_final_lock
        require_final_lock(root)
    roi_root = local_path(root, config["roi_root"], ("data", "raw", "CBIS-DDSM", "roi_cropped"))
    summary = {"status": "NOT_READY", "partition": partition, "protocol": config,
               "expected_roi_root": config["roi_root"], "expected_roi_root_absolute": str(roi_root.resolve()),
               "next_command": "python -m src.explainability.localisation --audit",
               "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    if not roi_root.is_dir() or not any(p.name != ".gitkeep" for p in roi_root.iterdir()):
        summary["reason"] = "ROI/cropped collection absent or empty; mapping and alignment not run; no localisation metrics"
        return summary, [], []
    images, provenance = selected_images(root, partition)
    lesions = annotation_references(root, images, partition)
    cases = {reference_parts(r[k])[0] for r in lesions for k in ("roi_mask_reference", "cropped_image_reference")}
    inventory = discover_annotations(roi_root, cases)
    table = map_annotations(lesions, inventory)
    locked_geometry = locked_cohort(root, coordinate_protocol, partition, table) if coordinate_protocol else {}
    development = {r['record_id']:r for r in read_rows(root/'reports/evidence/roi_alignment_development_checks.csv')} if coordinate_protocol and partition != 'test' else {}
    by_image = {r["canonical_image_id"]: r for r in images}
    review_path = local_path(root, config["alignment_reviews"], ("data", "interim"))
    reviews = read_rows(review_path) if review_path.exists() else []
    full_headers = {}
    full_hashes = {}
    for row in table:
        row.update(alignment_status="not_run", alignment_method="", mask_sha256="", geometry_sha256="", mask_metadata_json="", canonical_mask_pixels="")
        if row["role"] != "mask" or row["mapping_status"] != "unique":
            continue
        try:
            image = by_image[row["canonical_image_id"]]
            geometry = json.loads(image["preprocessing_metadata_json"])
            row["geometry_sha256"] = digest(geometry)
            full_path = local_path(root, image["dicom_relative_path"], ("data", "raw", "CBIS-DDSM", "images"))
            if image["canonical_image_id"] not in full_headers:
                full_headers[image["canonical_image_id"]] = pydicom.dcmread(full_path, stop_before_pixels=True)
            full_header = full_headers[image["canonical_image_id"]]
            verify_full_reference(image, full_header)
            if str(full_header.SOPInstanceUID) != geometry["SOPInstanceUID"]:
                raise ValueError("Full-image header differs from saved preprocessing metadata")
            mask_path = local_path(root, (roi_root / row["path"]).relative_to(root).as_posix(),
                                   ("data", "raw", "CBIS-DDSM", "roi_cropped"))
            decoded = decode_mask(mask_path, row["SOPInstanceUID"])
            row["mask_sha256"] = sha256_file(mask_path)
            row["mask_metadata_json"] = json.dumps(decoded.metadata, sort_keys=True)
            if coordinate_protocol:
                expected = locked_geometry[row['record_id']]
                shape = (int(full_header.Rows), int(full_header.Columns))
                if (shape != (int(expected['full_rows']),int(expected['full_columns']))
                        or decoded.mask.shape != (int(expected['mask_rows']),int(expected['mask_columns']))
                        or row['mask_sha256'] != expected['mask_sha256']):
                    raise ValueError('Mask geometry/hash differs from locked evidence')
                if partition != 'test':
                    evidence = development[row['record_id']]
                    if image['canonical_image_id'] not in full_hashes:
                        full_hashes[image['canonical_image_id']] = sha256_file(full_path)
                    if full_hashes[image['canonical_image_id']] != evidence['full_sha256']:
                        raise ValueError('Full mammogram changed since coordinate evidence')
                    crop = next(r for r in table if r['record_id']==row['record_id'] and r['role']=='cropped')
                    crop_path = local_path(root,(roi_root/crop['path']).relative_to(root).as_posix(),('data','raw','CBIS-DDSM','roi_cropped'))
                    if sha256_file(crop_path) != evidence['crop_sha256']:
                        raise ValueError('Supplied crop changed since coordinate evidence')
                rule = coordinate_rule(decoded.mask.shape, shape)
                if rule == 'EXCLUDE_UNPROVEN':
                    row.update(alignment_status='excluded',alignment_method=rule,reason='exact_spatial_transform_unproven')
                    continue  # Never resize or transform an unequal-grid annotation.
                # Retain the original rejection of contradictory known geometry.
                try:
                    prove_full_grid(decoded.header, full_header)
                except ValueError as exc:
                    if not str(exc).startswith('UNVERIFIED_ALIGNMENT:'):
                        raise
                row['alignment_method'] = 'locked_equal_grid_identity'
                aligned = transform_mask(decoded.mask, geometry)
                row.update(alignment_status='verified',canonical_mask_pixels=int(aligned.sum()))
                continue
            matches = [r for r in reviews if r.get("mask_sop_uid") == row["SOPInstanceUID"]]
            if len(matches) > 1:
                raise ValueError("Ambiguous manual alignment reviews")
            review = matches[0] if matches else None
            if review and (review.get("mask_sha256") != row["mask_sha256"] or review.get("geometry_sha256") != row["geometry_sha256"]):
                raise ValueError("Alignment review does not match current mask/geometry hashes")
            row["alignment_method"] = prove_full_grid(decoded.header, full_header, review)
            aligned = transform_mask(decoded.mask, geometry)
            row["alignment_status"] = "verified"
            row["canonical_mask_pixels"] = int(aligned.sum())
        except (OSError, ValueError, AttributeError, KeyError, pydicom.errors.InvalidDicomError) as exc:
            row.update(alignment_status="failed", reason=f"{type(exc).__name__}: {exc}")
    masks = [r for r in table if r["role"] == "mask"]
    # Repeated lesion references must not silently count the same annotation twice.
    keys = [(r["canonical_image_id"], r["abnormality_id"]) for r in masks]
    duplicate_lesions = len(keys) != len(set(keys))
    mask_paths = [r["path"] for r in masks if r["mapping_status"] == "unique"]
    duplicate_masks = len(mask_paths) != len(set(mask_paths))
    eligible = [r for r in masks if r['alignment_status']=='verified']
    excluded = [r for r in masks if r['alignment_status']=='excluded']
    ready = bool(eligible) and not duplicate_lesions and not duplicate_masks and all(r['alignment_status'] in ('verified','excluded') for r in masks)
    status = 'READY_FOR_VALIDATION_LOCALISATION' if ready and coordinate_protocol and partition=='validation' else ('READY' if ready else 'NOT_READY')
    eligible_ids = {r['canonical_image_id'] for r in eligible}
    eligible_images = [r for r in images if r['canonical_image_id'] in eligible_ids]
    multiplicities = Counter(r['canonical_image_id'] for r in eligible)
    summary.update(provenance, status=status, images=len(images), lesion_rows=len(lesions),
        eligible_masks=len(eligible), excluded_masks=len(excluded), eligible_images=len(eligible_images),
        eligible_participants=len({r['participant_id'] for r in eligible_images}),
        eligible_multiple_lesion_images=sum(n>1 for n in multiplicities.values()),
        eligible_lesions_per_image_histogram=dict(Counter(multiplicities.values())),
        image_union_definition='Union of eligible equal-grid lesions only; excluded annotations remain in audit table',
        final_test_localisation_ready=False if partition!='test' else ready,
        coordinate_protocol_sha256=bound_hash(root/PROTOCOL) if coordinate_protocol else None,
        scoped_inventory_files=len(inventory), unreadable_files=sum(r["read_status"] != "readable" for r in inventory),
        mask_mapping_counts=dict(Counter(r["mapping_status"] for r in masks)),
        alignment_counts=dict(Counter(r["alignment_status"] for r in masks)),
        duplicate_lesions=duplicate_lesions, duplicate_mask_assignments=duplicate_masks,
        reason="Eligible masks mapped and alignment verified; geometry exclusions recorded" if ready else "Missing/ambiguous masks or unverified alignment; metrics blocked",
        annotation_sha256={r["path"]: r["mask_sha256"] for r in masks if r["mask_sha256"]},
        reviews_sha256=sha256_file(review_path) if review_path.exists() else None,
        metadata_sha256=sha256_file(root / f"data/raw/CBIS-DDSM/metadata/mass_case_description_{'test' if partition == 'test' else 'train'}_set.csv"))
    return summary, table, eligible_images if ready else images


def write_audit(root: Path, summary: dict, table: list[dict]) -> None:
    """Write readiness evidence; mapping tables exist only when actually audited."""
    evidence = root / "reports/evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    if table:
        name = datetime.now(timezone.utc).strftime("roi_mapping_%Y%m%dT%H%M%S_%fZ.csv")
        fields = sorted({key for row in table for key in row})
        write_csv(evidence / name, table, fields)
        summary = dict(summary, mapping_audit_table=f"reports/evidence/{name}")
    (evidence / "roi_mapping_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
