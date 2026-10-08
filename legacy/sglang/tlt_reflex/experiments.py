"""Checkpoint provenance and experiment labels; setup/reporting only."""
import json
from pathlib import Path


def projector_record(draft):
    draft=Path(draft)
    cfg=json.loads((draft/'config.json').read_text()) if (draft/'config.json').is_file() else {}
    manifest=json.loads((draft/'conversion.json').read_text()) if (draft/'conversion.json').is_file() else {}
    provenance=cfg.get('opd_projector_provenance',manifest.get('opd_projector_provenance','unresolved'))
    training=cfg.get('opd_projector_training_metadata',manifest.get('opd_projector_training_metadata',{}))
    return dict(provenance=provenance,rank=cfg.get('opd_rank'),
        source_checkpoint=manifest.get('source_checkpoint'),training_dataset=training.get('dataset'),
        training_steps=training.get('steps'),training_metadata_status='recorded' if training else 'not_recorded',
        trained=provenance=='trained')


def experiment_label(method,provenance):
    if method=='tlt':return 'TLT'
    if provenance=='trained':return 'TLT + OPD(trained A + online B)'
    if provenance=='head_basis_initialized':return 'TLT + OPD(head-basis A + online B)'
    return 'TLT + OPD(unresolved A provenance)'
