"""Persistent compiler-capacity recovery; never overwrites parent evidence."""
import argparse
import os
import time
import traceback
from pathlib import Path

from medtoolbench.supplement.gptoss_compile_recovery import CONTROL,prepare,run
from medtoolbench.supplement.storage_recovery import atomic_json


def main():
    p=argparse.ArgumentParser(); p.add_argument('--control',type=Path,default=CONTROL)
    p.add_argument('--unit',type=Path,required=True); p.add_argument('--gpu-proof',type=Path,required=True)
    a=p.parse_args(); a.control=a.control.resolve(); a.control.mkdir(parents=True,exist_ok=True)
    with (a.control/'bootstrap.log').open('x') as log:
        os.dup2(log.fileno(),1); os.dup2(log.fileno(),2)
        try:
            prepare(a.control,a.unit.resolve(),a.gpu_proof.resolve())
            run(a.control/'manifest.json')
        except BaseException:
            atomic_json(a.control/'bootstrap_failure.json',{'error':traceback.format_exc(),'time':time.time()})
            raise


if __name__ == '__main__':
    main()
