"""Pure future image-level reporting utilities; no I/O or model calls."""
import numpy as np
from src.evaluation.metrics import evaluate_metrics
from .cmmd_provenance import THRESHOLD


def report_metrics(rows):
    """One verified full mammogram per SOP UID; cluster identifier is participant.

    Side/patient totals are descriptive. No unregistered probability aggregation
    across views, sides or participants is performed.
    """
    if not rows:
        raise ValueError('No predictions: external results cannot be fabricated')
    seen, side_labels = set(), {}
    for row in rows:
        uid = row['SOPInstanceUID']
        if not uid or uid in seen:
            raise ValueError('Missing/duplicate image UID')
        seen.add(uid)
        if not row['patient_id'] or row['laterality'] not in ('L', 'R') or row['label'] not in (0, 1):
            raise ValueError('Invalid participant/side/label')
        key = (row['patient_id'], row['laterality'])
        if key in side_labels and side_labels[key] != row['label']:
            raise ValueError('Conflicting breast-side labels')
        side_labels[key] = row['label']
    labels = np.array([r['label'] for r in rows])
    result = evaluate_metrics(labels, np.array([r['probability'] for r in rows]), THRESHOLD)
    result.update(number_images=len(rows), number_breast_sides=len(side_labels),
                  number_patients=len({r['patient_id'] for r in rows}),
                  benign_images=int(sum(labels == 0)), malignant_images=int(sum(labels == 1)),
                  benign_sides=sum(v == 0 for v in side_labels.values()),
                  malignant_sides=sum(v == 1 for v in side_labels.values()),
                  false_negative_count=result['fn'],
                  false_negative_rate=result['fn'] / int(sum(labels == 1)) if sum(labels == 1) else None,
                  analysis_unit='image', interpretation='Domain/generalisation findings; never model selection')
    return result


def participant_bootstrap(rows, iterations=2000, seed=42):
    """Percentile 95% CI; retain all images for every sampled participant draw."""
    report_metrics(rows)
    if type(iterations) is not int or iterations < 2:
        raise ValueError('At least two bootstrap iterations required')
    patients = sorted({r['patient_id'] for r in rows})
    if len(patients) < 2:
        raise ValueError('Participant bootstrap needs at least two participants')
    groups = {p: [r for r in rows if r['patient_id'] == p] for p in patients}
    keys = ('accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'balanced_accuracy',
            'roc_auc', 'pr_auc', 'brier_score', 'false_negative_rate')
    samples = {k: [] for k in keys}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        sample = [r for p in rng.choice(patients, len(patients), replace=True) for r in groups[p]]
        m = evaluate_metrics(np.array([r['label'] for r in sample]), np.array([r['probability'] for r in sample]), THRESHOLD)
        m['false_negative_rate'] = 1 - m['sensitivity'] if m['sensitivity'] is not None else None
        for key in keys:
            if m[key] is not None:
                samples[key].append(m[key])
    return dict(method='participant-cluster percentile bootstrap', seed=seed, iterations=iterations,
                intervals={k: {'ci95': np.quantile(v, [.025, .975]).tolist() if v else None,
                               'valid_replicates': len(v), 'undefined_replicates': iterations-len(v)} for k, v in samples.items()})
