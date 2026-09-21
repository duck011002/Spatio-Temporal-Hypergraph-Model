"""CPU-only archival snapshot; excludes secrets, raw data and checkpoints."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

root = Path('/root/autodl-tmp/STHGCN-git')
out = Path(tempfile.mkdtemp(prefix='paper_evidence_', dir='/root/autodl-tmp'))
def command(args):
    p = subprocess.run(args, cwd=root, capture_output=True, text=True)
    return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

packages = {}
for name in ['torch','torch-geometric','torch-sparse','torch-scatter','numpy','pandas','scipy','scikit-learn','PyYAML']:
    try: packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: packages[name] = None
snapshot = {'captured_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'CPU-mode post-run environment, not training hardware attestation',
    'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages,
    'git_head': command(['git','rev-parse','HEAD']),
    'git_status': command(['git','status','--porcelain']),
    'cpu': command(['lscpu']), 'memory': command(['free','-b']),
    'processes': command(['ps','-eo','pid,etime,stat,comm']),
    'disk': command(['df','-h','/root/autodl-tmp'])}
(out/'environment.json').write_text(json.dumps(snapshot, indent=2))
files = set()
for folder, patterns in {
    'runs':['**/*.json','**/*.csv'], 'remote_runs':['*.log'],
    'conf':['**/*.yml'], 'artifacts':['**/*.json','**/*.jsonl','**/*.npz'],
    'log':['**/train.log'], 'model':['*.py'], 'a5_jev':['*.py'],
    'utils':['pipeline_util.py','run_reporter.py'], 'rule_teacher':['*.py'],
}.items():
    for pattern in patterns: files.update(p for p in (root/folder).glob(pattern) if p.is_file())
for name in ['run.py','run_a5_jev.py','export_a5_candidates.py','freeze_a5_jev_safe.py','force_a5_test_after_rejected_validation.py']:
    if (root/name).exists(): files.add(root/name)
manifest = []
archive = out/'evidence.tgz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(files):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
        rel=p.relative_to(root).as_posix()
        manifest.append({'path':rel,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
        tar.add(p,arcname=rel)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    tar.add(out/'manifest.json',arcname='snapshot_manifest.json')
    tar.add(out/'environment.json',arcname='snapshot_environment.json')
print(json.dumps({'archive':str(archive),'files':len(files),'bytes':archive.stat().st_size,
                  'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}),flush=True)
