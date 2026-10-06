"""Hash local pinned model/kernel artifacts and record dependency provenance."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from medtoolbench.supplement.gptoss_model import REVISION
from medtoolbench.supplement.storage_recovery import atomic_json

root = Path(__file__).resolve().parents[2]
out = root / 'data/gptoss_replacement_v1/artifacts.json'
if out.exists():
    raise RuntimeError('Artifact receipt already exists')
model = root / 'data/models/gpt-oss-20b'
index = json.loads((model / 'model.safetensors.index.json').read_text())
files = sorted(set(index['weight_map'].values()) | {'config.json', 'generation_config.json',
    'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja', 'model.safetensors.index.json'})
records = {}
for name in files:
    file = model / name
    metadata = model / '.cache/huggingface/download' / (name + '.metadata')
    receipt = metadata.read_text().splitlines()
    if receipt[0] != REVISION:
        raise RuntimeError('Download revision differs: ' + name)
    with file.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    if name.endswith('.safetensors') and checksum != receipt[1]:
        raise RuntimeError('Weight SHA256 differs from Hub LFS receipt: ' + name)
    records[name] = {'size': file.stat().st_size, 'sha256': checksum, 'revision': receipt[0], 'etag': receipt[1]}
    print(json.dumps({'verified': name, 'bytes': file.stat().st_size}), flush=True)
kernel_root = root / 'data/gptoss_cache/hub/kernels--kernels-community--gpt-oss-triton-kernels'
kernel_files = {}
for file in sorted((kernel_root / 'snapshots').glob('**/*')):
    if file.is_file():
        with file.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        kernel_files[str(file.relative_to(kernel_root))] = checksum
if not kernel_files:
    raise RuntimeError('Kernel cache has not been downloaded')
atomic_json(out, {'all_pass': True, 'repo': 'openai/gpt-oss-20b', 'revision': REVISION,
    'model': str(model), 'files': records, 'kernel_root': str(kernel_root), 'kernel_files': kernel_files,
    'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
    'environment_note': 'Independent venv overrides, with shared read-only system packages; unrelated inherited package constraints are not claimed clean',
    'inference_api': False})
