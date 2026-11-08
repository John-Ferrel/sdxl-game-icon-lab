"""共享路径和内容指纹。仓库位置可以移动；机器路径只写在被忽略的local配置中。"""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1',
                  HF_HOME=str(ROOT/'runs/hf-cache'), CUBLAS_WORKSPACE_CONFIG=':4096:8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def settings():
    p=ROOT/'config.local.json'
    if not p.exists():
        raise RuntimeError('Copy config.example.json to config.local.json and fill your ComfyUI/checkpoint paths first.')
    return read(p)


def external_path(name):
    p=Path(settings()[name]).expanduser()
    return (p if p.is_absolute() else ROOT/p).resolve()


def checkpoint():
    p=external_path('checkpoint')
    if not p.is_file(): raise FileNotFoundError(f'SDXL checkpoint not found: {p}')
    return p


def comfyui():
    p=external_path('comfyui_dir')
    if not (p/'nodes.py').is_file(): raise FileNotFoundError(f'ComfyUI source not found: {p}')
    return p


LAB=ROOT/(settings().get('run_dir','runs/public-demo') if (ROOT/'config.local.json').exists() else 'runs/public-demo')
assert LAB.resolve().is_relative_to((ROOT/'runs').resolve()), 'Keep generated runs under this repository runs/'


def config():
    cfg=read(ROOT/'configs/train.json')
    cfg['manifest_sha256']=sha(ROOT/'data/manifest.json')
    cfg['base_checkpoint']=checkpoint().name
    return cfg


def local_environment():
    import importlib.metadata as m
    import platform
    return {'python':sys.version, 'platform':platform.platform(),
            'packages':{n:m.version(n) for n in ['torch','diffusers','peft','transformers',
                        'bitsandbytes','safetensors','Pillow','numpy']}}
