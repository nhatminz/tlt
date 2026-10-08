#!/usr/bin/env python3
"""Audit full source provenance, or check algorithm integrity before training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def is_runtime_source(destination):
    return destination.startswith('helper/') or destination in (
        'grpo_speculative.py', 'train_draft.py',
    )


def check(*, runtime=False):
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    errors = []
    checked = set()

    def verify(relative, expected, label):
        path = ROOT / relative
        if not path.is_file():
            errors.append(f'{label} missing: {relative}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f'{label} differs from manifest: {relative}')
        checked.add(relative)

    for entry in manifest['files']:
        destination = entry.get('destination') or ''
        if runtime and not is_runtime_source(destination):
            continue
        if not runtime:
            verify(entry['snapshot'], entry['sha256'], 'source snapshot')
        if destination:
            verify(destination, entry['port_sha256'], 'port')
            if entry['port_status'] == 'exact' and entry['port_sha256'] != entry['sha256']:
                errors.append('exact port differs from source: ' + destination)
    for name, expected in manifest.get('tlt_algorithm_files', {}).items():
        verify(name, expected, 'TLT implementation')
    if errors:
        mode = 'runtime integrity' if runtime else 'full provenance audit'
        hint = (
            'Sync the listed runtime files and SOURCE_MANIFEST.json from the same '
            'TltReflex revision. No hashes were rewritten.' if runtime else
            'For a full audit, sync sources/, local ports and SOURCE_MANIFEST.json '
            'from the same TltReflex revision. Training uses --runtime, which checks '
            'algorithm files without requiring archival snapshots or launcher hashes. '
            'After an intentional reviewed local adaptation, run '
            'python scripts/refresh_port_manifest.py and audit again.'
        )
        raise ValueError(f'{mode} failed:\n- ' + '\n- '.join(errors) + '\n' + hint)
    return len(checked) if runtime else len(manifest['files'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', action='store_true',
                        help='check helper/, GRPO/pretraining entrypoints and TLT algorithms; '
                             'omit archive snapshots and operational scripts/configs')
    args = parser.parse_args(argv)
    try:
        count = check(runtime=args.runtime)
    except (ValueError, OSError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        return 1
    if args.runtime:
        print(f'Runtime source integrity verified: {count} files '
              '(full provenance audit: python scripts/check_source_manifest.py)')
    else:
        print(f'Source manifest verified: {count} entries')
    return 0


if __name__ == '__main__':
    sys.exit(main())
