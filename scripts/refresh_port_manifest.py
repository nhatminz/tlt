#!/usr/bin/env python3
"""Update hashes of local adaptations; never rewrite source snapshot hashes."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    path=ROOT/'SOURCE_MANIFEST.json';manifest=json.loads(path.read_text())
    for entry in manifest['files']:
        if entry.get('destination'):
            entry['port_sha256']=hashlib.sha256((ROOT/entry['destination']).read_bytes()).hexdigest()
            entry['port_status']='exact' if entry['port_sha256']==entry['sha256'] else 'adapted'
    manifest['tlt_algorithm_files']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/'helper'/name for name in ('tlt_generate.py','tlt_scheduler.py','tlt_transition.py','tlt_workspace.py','tlt_mab.py')]}
    path.write_text(json.dumps(manifest,indent=2)+'\n')
