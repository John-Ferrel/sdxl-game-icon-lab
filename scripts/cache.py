"""第2步（GPU）：把冻结的图片/文字编码器输出保存，训练时不占它们的显存。

这里只编码训练集，不读取留出图片。VAE/CLIP没有被训练，缓存没有改变模型。
缓存绑定图像、caption、基础模型、分辨率与本脚本的哈希，防止误用旧内容。
"""
from common import ROOT, LAB, read, write, sha, config, checkpoint, comfyui
import hashlib
import json
import sys
import time
import subprocess


def main():
    cfg=config()
    manifest_path=ROOT/'data/manifest.json'
    assert sha(manifest_path)==cfg['manifest_sha256'], 'Dataset changed; version the experiment'
    rows=[r for r in read(manifest_path)['examples'] if r['split']=='train']
    for r in rows:
        assert sha(ROOT/r['image'])==r['image_sha256']
        assert sha(ROOT/r['caption_path'])==r['caption_sha256']
    identity={'manifest_sha256':sha(manifest_path),'base_sha256':cfg['base_sha256'],
              'resolution':cfg['resolution'],'cache_script_sha256':sha(__file__)}
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    index_path=LAB/'cache/index.json'
    if index_path.exists():
        old=read(index_path)
        assert old['fingerprint']==fingerprint, 'Cache definition changed; use a new experiment directory'
        assert all(sha(ROOT/r['path'])==r['sha256'] for r in old['entries'])
        print('SKIP: all cached examples match their hashes'); return
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                                     '--format=csv,noheader,nounits','--id=0'],text=True).strip())
    if free<6144:
        raise RuntimeError(f'Only {free} MiB free VRAM; close other GPU workloads before caching. No model loaded.')
    assert sha(checkpoint())==cfg['base_sha256'], 'Base model bytes changed'
    base={'runtime_args':cfg['runtime_args'],'model':{'filename':checkpoint().name}}
    sys.path.insert(0,str(comfyui()))
    sys.argv=[sys.argv[0]]+base['runtime_args']
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import nodes
    import folder_paths
    folder_paths.add_model_folder_path('checkpoints',str(checkpoint().parent))
    import comfy.model_management as mm
    import numpy as np
    from PIL import Image
    from safetensors.torch import save_file
    torch.manual_seed(cfg['seed'])
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    index={'identity':identity,'fingerprint':fingerprint,'entries':[],
           'scope':'training images only; frozen CLIP/VAE; no augmentation', 'status':'running'}
    try:
        # 复用固定的SDXL checkpoint和ComfyUI编码路径。
        model,clip,vae=nodes.CheckpointLoaderSimple().load_checkpoint(base['model']['filename'])
        del model  # 此进程不采样图片，不需要保留UNet引用。
        encoded=[]
        with torch.inference_mode():
            for row in rows:
                cond=nodes.CLIPTextEncode().encode(clip,row['caption'])[0][0]
                emb=cond[0].cpu().float().contiguous()
                pool=cond[1]['pooled_output'].cpu().float().contiguous()
                # 本课短caption应处于一个CLIP文本窗口内；超长时不静默截断。
                assert emb.shape[1]==77, f'Caption exceeds one window: {row["id"]}'
                encoded.append((emb,pool))
                print('TEXT',row['id'],list(emb.shape),flush=True)
            mm.unload_all_models()
            del clip
            for row,(emb,pool) in zip(rows,encoded):
                with Image.open(ROOT/row['image']) as image:
                    assert image.mode=='RGB' and image.size==(512,512)
                    pixels=torch.from_numpy(np.array(image).copy()).float()[None]/255.
                raw=nodes.VAEEncode().encode(vae,pixels)[0]['samples'].cpu().float()
                # ComfyUI VAE返回未缩放latent；Diffusers的SDXL训练使用0.13025缩放。
                tensors={'latents':(raw*0.13025).contiguous(),
                         'prompt_embeds':emb,'pooled_prompt_embeds':pool,
                         # 原始尺寸、裁切起点、目标尺寸；本课没有裁切或改变尺寸。
                         'time_ids':torch.tensor([[512,512,0,0,512,512]],dtype=torch.float32)}
                assert tensors['latents'].shape==(1,4,64,64)
                assert all(torch.isfinite(v).all() for v in tensors.values())
                dst=LAB/'cache'/f'{row["id"]}.safetensors'
                temp=dst.with_suffix('.tmp')
                save_file(tensors,str(temp),metadata={'fingerprint':fingerprint,
                          'image_sha256':row['image_sha256'], 'caption_sha256':row['caption_sha256']})
                temp.replace(dst)
                index['entries'].append({'id':row['id'],'path':str(dst.relative_to(ROOT)),
                                         'sha256':sha(dst),'shapes':{k:list(v.shape) for k,v in tensors.items()}})
                print('IMAGE',row['id'],flush=True)
        index['status']='success'
        write(index_path,index)
    except Exception as e:
        index.update(status='failed',error=repr(e))
        raise
    finally:
        index.update(wall_seconds=time.perf_counter()-started,
                     peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_torch_reserved_bytes=torch.cuda.max_memory_reserved())
        write(LAB/'logs/cache.json',index)
        print(json.dumps({'status':index['status'],'examples':len(index['entries']),
                          'wall_seconds':index['wall_seconds']}),flush=True)


if __name__=='__main__': main()
