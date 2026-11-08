"""启动独立的ComfyUI阅读/练习界面；不提交生成任务。"""
from common import ROOT, LAB, config, comfyui, checkpoint, write, settings
import argparse
import json
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8190)
    args=p.parse_args();cfg=config()
    # 旧版ComfyUI在解析参数时就检查用户目录，首次启动前必须先建立。
    for folder in ['ui-user','ui-temp','outputs']:
        (LAB/folder).mkdir(parents=True,exist_ok=True)
    # YAML的字符串用JSON引号编码，避免Windows路径中的特殊字符产生歧义。
    paths={'checkpoints':checkpoint().parent,'loras':LAB/'checkpoints'}
    if settings().get('controlnet'):paths['controlnet']=ROOT/settings()['controlnet']
    text='public_lab:\n'
    for key,value in paths.items():
        if key=='controlnet':value=value.resolve().parent
        text+=f'  {key}: {json.dumps(value.as_posix())}\n'
    path=LAB/'model-paths.yaml';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8')
    cmd=[sys.executable,'-B',str(comfyui()/'main.py'),*cfg['runtime_args'],
         '--listen','127.0.0.1','--port',str(args.port),'--extra-model-paths-config',str(path),
         '--input-directory',str(ROOT/'data/input'),'--output-directory',str(LAB/'outputs'),
         '--user-directory',str(LAB/'ui-user'),'--temp-directory',str(LAB/'ui-temp')]
    print(f'Open http://127.0.0.1:{args.port}; drag a JSON from workflows/.',flush=True)
    raise SystemExit(subprocess.call(cmd,cwd=ROOT))


if __name__=='__main__':main()
