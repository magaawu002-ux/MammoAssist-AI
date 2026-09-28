"""One explicitly selected Validation canonical image through the app's model path."""

from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from src.app.core import contract,load_model,analyse,CHECKPOINT,THRESHOLD
from src.app.export import png_report,pdf_report
from src.data.provenance import sha256_file
from src.preprocessing.manifests import read_rows,local_path
from src.preprocessing.model_inputs import transfer_input
from src.utils.paths import PROJECT_ROOT


def run(root=PROJECT_ROOT):
    destination=root/'reports/evidence/app'
    if (destination/'app_smoke.json').exists(): raise ValueError('Existing app smoke evidence is preserved; refusing overwrite')
    details=contract(root,verify_checkpoint=True)
    manifest=root/'data/processed/manifests/validation_processed.csv'
    preprocessing=json.loads((root/'reports/evidence/preprocessing_validation.json').read_text())
    if sha256_file(root/'data/splits/validation.csv')!=preprocessing['split_sha256']['validation']:
        raise ValueError('Validation split changed')
    rows=read_rows(manifest)
    if rows!=read_rows(root/'reports/evidence/preprocessing_validation_attempt.csv'):
        raise ValueError('Validation manifest changed')
    if any(r['final_partition']!='validation' or r['official_partition']!='train' for r in rows):
        raise ValueError('Validation-only smoke required')
    row=min(rows,key=lambda r:r['canonical_image_id'])
    geometry=json.loads(row['preprocessing_metadata_json'])
    if row['mapping_status']!='uniquely_mapped' or geometry['clahe'] is not False:
        raise ValueError('Unverified canonical image')
    cache=local_path(root,row['cache_relative_path'],('data','processed','cache'))
    if sha256_file(cache)!=row['cache_sha256']: raise ValueError('Cache changed')
    image=np.load(cache,allow_pickle=False)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(details['config']['random_seed'])
    tf.config.experimental.enable_op_determinism()
    resource=load_model(root); result=analyse(image,resource)
    ordinary=float(resource.explainer.model(transfer_input(image,'resnet50')[None,...],training=False).numpy()[0,0])
    if abs(ordinary-result['score'])>1e-6: raise ValueError('App Grad-CAM score differs from ordinary inference')
    if sha256_file(root/CHECKPOINT)!=details['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    timestamp=datetime.now(timezone.utc).isoformat(timespec='seconds')
    # Deterministic research reference; no user upload is persisted by this separate evidence command.
    reference='upload-000000000001'
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'validation_smoke_summary.png').write_bytes(png_report(result,image,reference,timestamp,[]))
    (destination/'validation_smoke_summary.pdf').write_bytes(pdf_report(result,image,reference,timestamp,[]))
    evidence=dict(status='PASS',partition='validation',canonical_image_id=row['canonical_image_id'],
        selection_rule='lexicographically smallest Validation canonical image ID',
        checkpoint=CHECKPOINT,checkpoint_sha256=details['checkpoint_sha256'],cache_sha256=row['cache_sha256'],
        model_score=result['score'],ordinary_model_score=ordinary,score_absolute_difference=abs(ordinary-result['score']),
        threshold=THRESHOLD,target_layer=details['target_layer'],gradcam_status='PASS',
        gradcam_finite=bool(np.isfinite(result['heatmap']).all()),gradcam_shape=list(result['heatmap'].shape),
        gradcam_range=[float(result['heatmap'].min()),float(result['heatmap'].max())],zero_heatmap=result['zero_heatmap'],
        preprocessing_version='canonical_v1',clahe=False,no_test_image_or_prediction_access=True,
        no_evaluation_or_training_invoked=True,checkpoint_unchanged=True,timestamp=timestamp,
        evidence_type='Actual app-core inference and export, not a browser screenshot or clinical validation',
        source_sha256={p.name:sha256_file(p) for p in sorted((root/'src/app').glob('*.py'))},
        artifacts_sha256={p.name:sha256_file(p) for p in sorted(destination.glob('validation_smoke_summary.*'))})
    with (destination/'app_smoke.json').open('x',encoding='utf-8') as f:json.dump(evidence,f,indent=2,allow_nan=False);f.write('\n')
    return evidence


if __name__=='__main__': print(json.dumps(run(),indent=2))
