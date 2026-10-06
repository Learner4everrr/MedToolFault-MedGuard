"""Persistent post-build bootstrap; no formal runtime/package mutation."""
import argparse
import os
import subprocess
import time
import traceback
from pathlib import Path

from medtoolbench.supplement import qwen_kernel_probe as probe
from medtoolbench.supplement.qwen_kernel_window import identity, same_process, validate_unit
from medtoolbench.supplement.storage_recovery import atomic_json, read


ENVIRONMENT_CHECK = '''
import hashlib, importlib.metadata as metadata, json, pathlib, sys
import torch, transformers, fla, causal_conv1d, causal_conv1d_cuda
from transformers.models.qwen3_5 import modeling_qwen3_5 as q
from medtoolbench.supplement.qwen_kernel_probe import implementations
names = ('causal_conv1d_fn','causal_conv1d_update','torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule')
bindings = {n: implementations(getattr(q,n)) for n in names}
assert all(len(v)==1 and v[0]['optimized'] for v in bindings.values()), bindings
def checksum(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
paths = [q.__file__, causal_conv1d_cuda.__file__]
paths += [str(f) for m in (fla,causal_conv1d) for f in pathlib.Path(m.__file__).parent.rglob('*.py')]
print(json.dumps({'python':sys.executable,'torch':torch.__version__,'transformers':transformers.__version__,
    'versions':{n:metadata.version(n) for n in ('flash-linear-attention','fla-core','causal-conv1d')},
    'bindings':bindings,'package_file_sha256':{p:checksum(p) for p in sorted(paths)}}))
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--build-pid', type=int, required=True)
    p.add_argument('--control', type=Path, default=probe.CONTROL)
    args = p.parse_args()
    control = args.control.resolve()
    with (control/'bootstrap.log').open('x') as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            m, plan, _ = probe.load(control)
            validate_unit(control)
            process = identity(args.build_pid)
            if process and process['state'] != 'Z':
                if 'env_qwen_fast/bin/python -m pip install ' not in process['cmdline'] or 'causal-conv1d==1.6.0' not in process['cmdline']:
                    raise RuntimeError('Unexpected build process')
                deadline = time.time()+1800
                while same_process(process):
                    if time.time()>deadline:
                        raise RuntimeError('Source build did not finish within the diagnostic bootstrap budget')
                    atomic_json(control/'status.json', {'state':'waiting_source_build','time':time.time(),
                        'formal_dispatch_running':True,'build_pid':args.build_pid})
                    time.sleep(10)
            if not read(control/'causal_source_install.json'):
                raise RuntimeError('Successful local source installation receipt missing')
            checked = subprocess.run([m['fast_python'],'-c',ENVIRONMENT_CHECK],
                capture_output=True,text=True,timeout=180)
            with (control/'environment_check.stderr').open('x') as stream:
                stream.write(checked.stderr)
            if checked.returncode:
                raise RuntimeError('Fast environment import/dispatch check failed: '+checked.stderr)
            import json
            record = json.loads(checked.stdout.strip().splitlines()[-1])
            if record['torch'] != m['expected_versions']['torch'] or record['transformers'] != m['expected_versions']['transformers']:
                raise RuntimeError('Baseline numerical runtime changed')
            for name, version in record['versions'].items():
                if version != m['expected_versions'][name]:
                    raise RuntimeError('Unexpected kernel version')
            atomic_json(control/'environment.json', record)
            atomic_json(control/'bootstrap_ready.json', {'time':time.time(),'all_pass':True,
                'formal_environment_unchanged':True,'model_inference_performed':False})
            os.execv(plan['python'], [plan['python'],'-u','-m','medtoolbench.supplement.qwen_kernel_window',
                'run','--control',str(control)])
        except BaseException:
            atomic_json(control/'bootstrap_failure.json', {'error':traceback.format_exc(),'time':time.time()})
            raise


if __name__ == '__main__':
    main()
