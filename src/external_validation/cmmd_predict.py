"""Explicit future CMMD prediction command; importing this module does not load a model."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from . import cmmd_conversion as amendment
from . import cmmd_image_audit as audit
from .cmmd_provenance import ROOT, CHECKPOINT, THRESHOLD, RESULTS, isolated_path, write_new, sha256

PREDICTIONS = f'{RESULTS}/predictions/cmmd_image_predictions_v2.csv'
RECEIPT = f'{RESULTS}/predictions/cmmd_inference_attempt_v2.json'


def execute(root=ROOT, *, confirmed=False):
    if confirmed is not True: raise ValueError('Explicit --execute is required')
    root=Path(root).resolve()
    lock=amendment.verify(root)
    amendment.no_external_predictions(root)
    for relative in (PREDICTIONS,RECEIPT):
        if isolated_path(relative,root).exists(): raise FileExistsError(relative)
    rows=audit.csv_read(root/amendment.COHORT)
    config=yaml.safe_load((root/'config/project.yaml').read_text())
    write_new(RECEIPT,json.dumps(dict(status='STARTED',protocol_sha256=sha256(root/amendment.PROTOCOL),
        checkpoint_sha256=lock['checkpoint_sha256'],threshold=THRESHOLD))+'\n',root)
    # Lazy imports occur only after all locks, identities and explicit authorization pass.
    from tensorflow.keras.models import load_model
    from src.preprocessing.canonical import preprocess_dicom
    from src.preprocessing.model_inputs import transfer_input
    model=load_model(root/CHECKPOINT,compile=False)
    model.trainable=False
    if model.input_shape!=(None,224,224,3) or model.output_shape!=(None,1):
        raise ValueError('Unexpected locked model signature')
    predictions=[]
    for row in rows:
        path=audit._safe_local(root,row['path'],audit.IMAGES)
        if sha256(path)!=row['sha256']:raise ValueError('Input changed after verification')
        result=preprocess_dicom(path,config)
        if audit.validate_preprocessed(result)!=row['array_sha256']:raise ValueError('Canonical output changed')
        image=transfer_input(result.image,'resnet50')
        output=np.asarray(model(image[np.newaxis,...],training=False))
        if output.shape!=(1,1) or not np.isfinite(output).all() or not 0<=float(output[0,0])<=1:
            raise ValueError('Invalid prediction')
        probability=float(output[0,0])
        predictions.append(dict(patient_id=row['patient_id'],laterality=row['laterality'],view=row['view'],
            SOPInstanceUID=row['sop_uid'],label=int(row['label']),probability=probability,
            predicted_label=int(probability>=THRESHOLD),threshold=THRESHOLD))
    amendment.verify(root)
    write_new(PREDICTIONS,audit.csv_bytes(predictions,list(predictions[0])).decode(),root)
    # Keep the attempt receipt immutable; a separate completion receipt marks success.
    write_new(f'{RESULTS}/predictions/cmmd_inference_complete_v2.json',json.dumps(dict(status='COMPLETE',
        images=len(predictions),predictions_sha256=sha256(root/PREDICTIONS),threshold=THRESHOLD))+'\n',root)
    return len(predictions)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Explicitly authorize prediction after all protocol guards pass')
    args=parser.parse_args()
    print(execute(confirmed=args.execute))
