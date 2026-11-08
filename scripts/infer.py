"""第4步：生成一张图或小对照组。每项独立进程，已完成项校验后跳过。

示例：python scripts/infer.py --lora 80 --width 768 --height 512
宽高在这里以及ComfyUI的EmptyLatentImage中可改；LoRA不会把输出锁死为512。
"""
from common import ROOT, LAB, config, checkpoint, sha, write, read
from pathlib import Path
import argparse
import subprocess
import sys

PROMPTS=[('compass','iconstyle, a single brass compass with a blue glass center, on a light gray background'),
         ('backpack','iconstyle, a single blue canvas backpack with a top handle and one front pocket, on a light gray background')]
NEGATIVE='text, watermark, logo, collage, multiple views, cropped, blurry, malformed'


def make_job(name,prompt,seed,lora,width=512,height=512,strength=1.0):
    if width%64 or height%64 or min(width,height)<256 or max(width,height)>1024:
        raise ValueError('Use multiples of64 between256 and1024; larger sizes are not guaranteed to fit8GB.')
    cfg=config(); graph={'1':{'class_type':'CheckpointLoaderSimple','inputs':{'ckpt_name':checkpoint().name}}}
    model,clip=['1',0],['1',1]; adapter=None
    if lora!='base':
        path=LAB/'checkpoints'/f'{cfg["experiment_id"]}_seed{cfg["seed"]}_step{int(lora):04d}.safetensors'
        if not path.exists():raise FileNotFoundError(f'Train first, or use --lora base: {path}')
        graph['2']={'class_type':'LoraLoader','inputs':{'model':model,'clip':clip,'lora_name':path.name,
                     'strength_model':strength,'strength_clip':0.0}}
        model,clip=['2',0],['2',1]
        adapter={'name':path.name,'sha256':sha(path),'strength_model':strength,'strength_clip':0.0}
    graph.update({
      '3':{'class_type':'CLIPTextEncode','inputs':{'text':prompt,'clip':clip}},
      '4':{'class_type':'CLIPTextEncode','inputs':{'text':NEGATIVE,'clip':clip}},
      '9':{'class_type':'EmptyLatentImage','inputs':{'width':width,'height':height,'batch_size':1}},
      '10':{'class_type':'KSampler','inputs':{'model':model,'seed':seed,'steps':25,'cfg':7.0,
            'sampler_name':'euler','scheduler':'normal','positive':['3',0],'negative':['4',0],
            'latent_image':['9',0],'denoise':1.0}},
      '11':{'class_type':'VAEDecodeTiled','inputs':{'samples':['10',0],'vae':['1',2],'tile_size':512,
            'overlap':64,'temporal_size':64,'temporal_overlap':8}},
      '12':{'class_type':'SaveImage','inputs':{'images':['11',0],'filename_prefix':f'{name}_seed{seed}'}}})
    return {'id':name,'stage':LAB.relative_to(ROOT).as_posix(),'seed':seed,'resolution':[width,height],
            'steps':25,'cfg':7.0,'sampler':'euler','scheduler':'normal','denoise':1.0,
            'mode':'txt2img','positive':prompt,'negative':NEGATIVE,
            'purpose':'CC0-only public demo; style observation, not a quality benchmark',
            'base_model':{'repo':'stabilityai/stable-diffusion-xl-base-1.0',
                          'revision':'462165984030d82259a11f4367a4eed129e94a7b',
                          'sha256':cfg['base_sha256'],'filename':checkpoint().name},
            'controlnet':None,'lora':adapter,'workflow':graph}


def run(job):
    key=f'{job["id"]}_seed{job["seed"]}'; path=LAB/'workflows'/f'{key}.job.json'
    if path.exists() and read(path)!=job:
        raise RuntimeError('Experiment ID already used with different parameters; choose --name NEW_NAME')
    write(path,job)
    with (LAB/'logs'/f'{key}.console.log').open('a',encoding='utf-8') as f:
        completed=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/native.py'),str(path)],
                                  cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if completed.returncode:raise RuntimeError(f'Inference failed; see runs/public-demo/logs/{key}.console.log')
    print('Finished',key,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',action='store_true')
    p.add_argument('--lora',choices=['base','40','80'],default='80')
    p.add_argument('--width',type=int,default=512);p.add_argument('--height',type=int,default=512)
    p.add_argument('--seed',type=int,default=20261026)
    p.add_argument('--strength',type=float,default=1.0)
    p.add_argument('--prompt',default=PROMPTS[0][1])
    p.add_argument('--name',default='PUBLIC-PRACTICE')
    a=p.parse_args()
    if a.suite:
        for subject,prompt in PROMPTS:
            for seed in [20261026,20261027]:
                for arm in ['base','40','80']:
                    run(make_job(f'PUBLIC-{subject}-{arm}',prompt,seed,arm))
    else:run(make_job(a.name,a.prompt,a.seed,a.lora,a.width,a.height,a.strength))


if __name__=='__main__':main()
