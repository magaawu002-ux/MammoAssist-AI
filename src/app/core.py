"""In-memory research app logic; no cohort evaluation, training or upload persistence."""

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import re
import threading

import numpy as np
from PIL import Image
import pydicom

from src.data.provenance import sha256_file
from src.explainability.coordinate_protocol import bound_hash
from src.explainability.overlay import overlay_heatmap
from src.preprocessing.canonical import preprocess_dicom, crop_background, resize_and_pad
from src.preprocessing.manifests import read_rows
from src.preprocessing.settings import preprocessing_settings
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT

TITLE = 'MammoAssist AI — Research Decision-Support Prototype'
DISCLAIMER = 'Not a medical device. Not for autonomous diagnosis. Outputs require qualified clinician interpretation.'
SCORE_NOTE = 'The model score is not a calibrated clinical probability of cancer.'
ATTENTION_NOTE = ('Grad-CAM is a model-attention visualisation, not a lesion detector or segmentation result. '
                  'Highlighted regions indicate areas that influenced the model output. They do not represent a validated lesion boundary.')
DOMAIN_NOTE = ('Trained and evaluated on CBIS-DDSM digitised film mammograms. Performance may not generalise '
               'to other equipment or populations. These are research results, not clinical validation.')
CHECKPOINT = 'results/experiments/resnet50/20260925T083032_895231Z/best_model.keras'
CHECKPOINT_HASH = '0076686617cb29ad5e5682f3d1eff89fae9cad2cebffa86e2b8be2b537934384'
THRESHOLD = 0.3753475248813629
TARGET = 'conv5_block3_out'
MAX_BYTES = 128 * 1024 * 1024
MAX_PIXELS = 32_000_000


class AppError(ValueError):
    """A sanitised message suitable for the app; never include upload metadata."""


def contract(root=PROJECT_ROOT, *, verify_checkpoint=False):
    """Check model/preprocessing bindings without reading any cohort pixels or predictions."""
    try:
        selection=json.loads((root/'results/experiments/final_selection.json').read_text())
        lock=json.loads((root/'results/explainability/localisation_lock.json').read_text())
        coordinate=json.loads((root/'reports/evidence/roi_alignment_protocol.json').read_text())
        config=load_config(root/'config/project.yaml'); settings=preprocessing_settings(config)
        preprocessing=json.loads((root/'reports/evidence/preprocessing_validation.json').read_text())
        identity=lock['identity']
        if (selection['run']+'/best_model.keras'!=CHECKPOINT or selection['checkpoint_sha256']!=CHECKPOINT_HASH
                or selection['threshold']!=THRESHOLD or lock['classifier_threshold']!=THRESHOLD
                or identity['checkpoint']!=CHECKPOINT or identity['checkpoint_sha256']!=CHECKPOINT_HASH
                or lock['target_layer']!=TARGET or lock['heatmap_threshold']!=.5
                or identity['classifier_selection_sha256']!=sha256_file(root/'results/experiments/final_selection.json')
                or identity['coordinate_protocol_sha256']!=bound_hash(root/'reports/evidence/roi_alignment_protocol.json')
                or settings['input_size']!=224 or settings['version']!='canonical_v1'
                or settings['crop_background'] is not True or settings['crop_margin_fraction']!=.02):
            raise ValueError('Contract mismatch')
        if preprocessing['settings']!=settings or preprocessing['status']!='COMPLETE':
            raise ValueError('Training preprocessing settings changed')
        for name in ('model_inputs.py','settings.py'):
            if sha256_file(root/'src/preprocessing'/name)!=preprocessing['implementation']['source_sha256'][name]:
                raise ValueError('Model input adapter changed')
        for name in ('gradcam.py','overlay.py'):
            if sha256_file(root/'src/explainability'/name)!=identity['implementation_sha256'][name]:
                raise ValueError('Explanation source changed')
        for name in ('canonical','dicom','manifests'):
            path=f'src/preprocessing/{name}.py'
            if bound_hash(root/path)!=coordinate['implementation_sha256'][path]:
                raise ValueError('Preprocessing source changed')
        if verify_checkpoint and sha256_file(root/CHECKPOINT)!=CHECKPOINT_HASH:
            raise ValueError('Checkpoint changed')
        return dict(architecture='ResNet50',checkpoint=CHECKPOINT,checkpoint_sha256=CHECKPOINT_HASH,
                    threshold=THRESHOLD,target_layer=TARGET,input_size=224,preprocessing_version='canonical_v1',
                    heatmap_research_threshold=.5,clahe=False,config=config)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise AppError('The locked model configuration or files could not be verified. Ask the project maintainer to check the installation.') from exc


def annotation_name(value):
    text=str(value).lower()
    return bool(re.search(r'(^|[^a-z])(roi|mask|cropped|crop|segmentation)([^a-z]|$)',text)
                or re.search(r'mass-(training|test)_p_\d+_(left|right)_(cc|mlo)_\d+',text))


@dataclass
class Upload:
    original: np.ndarray
    canonical: np.ndarray
    technical: dict
    warnings: list[str]
    quality: dict


def quality_checks(image):
    """Display-only input heuristics; no clinical quality certification or score correction."""
    h,w=image.shape; mean=float(np.mean(image)); contrast=float(np.std(image)); warnings=[]
    if min(h,w)<224: warnings.append('Severe resolution limitation: shorter dimension is below 224 pixels; upsampling cannot restore detail.')
    elif min(h,w)<1024: warnings.append('Resolution is low compared with typical full mammograms.')
    if mean<.05: warnings.append('Image is mostly dark under the locked normalisation; inspect the input.')
    elif mean>.95: warnings.append('Image is mostly bright under the locked normalisation; inspect the input.')
    if contrast<.03: warnings.append('Severe low-contrast input: the model cannot compensate for missing image detail.')
    return dict(height=h,width=w,normalised_mean=mean,normalised_std=contrast,warnings=warnings,
                interpretation='Input-quality heuristics only; not clinical image-quality certification.')


def prepare_upload(data, filename, config):
    """Decode bytes only; metadata is allowlisted and original names are never retained."""
    if not data or len(data)>MAX_BYTES: raise AppError('Upload an image no larger than 128 MiB.')
    if annotation_name(filename): raise AppError('ROI masks and cropped-lesion files are not accepted. Upload a full mammogram.')
    extension=Path(filename).suffix.lower(); warnings=[]
    try:
        settings=preprocessing_settings(config)
        if settings['input_size']!=224 or settings['version']!='canonical_v1' or not settings['crop_background'] or settings['crop_margin_fraction']!=.02:
            raise AppError('Preprocessing must match the locked canonical_v1 configuration.')
        if extension in ('.dcm','.dicom'):
            header=pydicom.dcmread(BytesIO(data),stop_before_pixels=True)
            descriptors=' '.join(str(getattr(header,k,'')) for k in ('SeriesDescription','StudyDescription','ImageComments','ImageType','PatientID'))
            if annotation_name(descriptors): raise AppError('DICOM metadata identifies an ROI/cropped annotation. Only full mammograms are accepted.')
            if str(getattr(header,'Modality',''))!='MG': raise AppError('Only single-frame mammography (MG) DICOM images are supported.')
            h,w=int(header.Rows),int(header.Columns)
            if h<=0 or w<=0 or h*w>MAX_PIXELS: raise AppError('Image dimensions are invalid or exceed the 32-megapixel limit.')
            result=preprocess_dicom(BytesIO(data),config,keep_stages=True)
            original=result.decoded.astype(np.float32); canonical=result.image
            technical={key:result.metadata[key] for key in ('original_height','original_width','photometric_interpretation',
                       'inverted','BitsStored','preprocessing_version','input_size','crop_reason')}
            technical['format']='DICOM'; technical['clahe']=False
            raw_range=None
        elif extension in ('.png','.jpg','.jpeg'):
            with Image.open(BytesIO(data)) as source:
                if source.format!=('PNG' if extension=='.png' else 'JPEG') or getattr(source,'n_frames',1)!=1:
                    raise AppError('Image format does not match its extension, or contains multiple frames.')
                w,h=source.size
                if h*w>MAX_PIXELS or min(h,w)<=0: raise AppError('Image exceeds the 32-megapixel limit.')
                if source.mode in ('RGBA','LA','P') or source.getexif().get(274,1)!=1:
                    raise AppError('Transparent/palette or orientation-tagged images are unsupported. Supply an upright grayscale full mammogram.')
                if source.mode not in ('L','I;16','I;16B','I','RGB'): raise AppError('Unsupported image colour mode.')
                if source.mode=='RGB': warnings.append('Colour image converted to grayscale; confirm this is a full mammogram, not a photograph or screenshot.')
                raw=np.asarray(source.convert('L') if source.mode=='RGB' else source,dtype=np.float32)
                scale=65535 if source.mode in ('I;16','I;16B','I') else 255
            if raw.ndim!=2 or not np.isfinite(raw).all() or raw.max()==raw.min(): raise AppError('Image has no usable finite grayscale intensity range.')
            raw_range=float(np.ptp(raw)/scale)
            original=(raw-raw.min())/(raw.max()-raw.min())
            cropped,_=crop_background(original,settings['crop_background'],settings['crop_margin_fraction'])
            canonical,_=resize_and_pad(cropped,224); canonical=np.clip(canonical,0,1).astype(np.float32)
            technical=dict(format='PNG' if extension=='.png' else 'JPEG',original_height=h,original_width=w,
                           preprocessing_version='canonical_v1',input_size=224,clahe=False)
            warnings.append('PNG/JPEG lacks DICOM metadata and acquisition provenance; the grayscale normalisation may not reproduce the original DICOM input.')
            if raw_range<.05: warnings.append('Severe limited source intensity range; normalisation can amplify noise rather than recover detail.')
        else: raise AppError('Unsupported format. Use DICOM (.dcm/.dicom), PNG or JPEG.')
        # Reject obvious two-level annotations even if renamed; no attempt to infer a lesion crop from anatomy.
        if np.unique(original).size<=2: raise AppError('Binary/constant images may be ROI masks and cannot be used as full mammograms.')
        quality=quality_checks(original)
        return Upload(original,canonical,technical,warnings+quality['warnings'],quality)
    except AppError: raise
    except Exception as exc:
        raise AppError('The image could not be decoded or preprocessed. Check for valid single-frame grayscale pixel data, supported MONOCHROME1/2 encoding and installed DICOM decoders.') from exc


class LockedModel:
    def __init__(self,explainer):
        self.explainer=explainer
        self.mutex=threading.Lock()


def load_model(root=PROJECT_ROOT):
    contract(root,verify_checkpoint=True)
    try:
        from src.explainability.gradcam import ResNet50GradCAM
        return LockedModel(ResNet50GradCAM.from_checkpoint(root/CHECKPOINT,input_size=224))
    except Exception as exc:
        raise AppError('The locked ResNet50 model could not be loaded. Check the project environment and checkpoint installation.') from exc


def analyse(image,resource):
    try:
        with resource.mutex:
            explanation=resource.explainer.explain(image)
        score=float(explanation.malignant_probability)
        if not np.isfinite(score) or not 0<=score<=1 or explanation.target_layer!=TARGET:
            raise ValueError('Invalid model result')
        overlay=overlay_heatmap(image,explanation.heatmap)
        return dict(score=score,threshold=THRESHOLD,decision=score_wording(score),
                    heatmap=explanation.heatmap,overlay=overlay,zero_heatmap=explanation.zero_heatmap)
    except Exception as exc:
        raise AppError('Model inference or genuine Grad-CAM failed. No substitute attention map has been generated.') from exc


def score_wording(score):
    if not np.isfinite(score) or not 0<=score<=1: raise AppError('Invalid model score.')
    return 'Output is '+('above or equal to' if score>=THRESHOLD else 'below')+' the prespecified research threshold.'


def performance(root=PROJECT_ROOT):
    """Read committed aggregate evidence only; never load predictions, images or evaluators."""
    base=root/'reports/evidence/model_evaluation'
    row=next(r for r in read_rows(base/'resnet50_validation_vs_test.csv') if r['partition']=='OFFICIAL HOLDOUT TEST')
    n=next(r for r in read_rows(base/'cohort_counts.csv') if r['partition']=='test')
    local=json.loads((root/'reports/evidence/final_localisation/summary.json').read_text())
    if float(row['threshold'])!=THRESHOLD or local['heatmap_threshold']!=.5: raise AppError('Research evidence threshold mismatch.')
    return dict(images=int(n['images']),metrics={k:float(row[k]) for k in ('sensitivity','specificity','balanced_accuracy','roc_auc','pr_auc','brier_score')},
                localisation=local['groups']['lesion/overall'])
