"""第3步：只训练SDXL UNet中的LoRA；基础模型、VAE和文字编码器不训练。

核心算法参考Diffusers v0.32.2 SDXL LoRA范例；冻结编码器缓存以适应8GB显存。
这次新增：真实多图采样、独立缓存校验、逐步样本记录、可续跑的数据顺序。
源码参考（Apache-2.0）：
https://github.com/huggingface/diffusers/blob/v0.32.2/examples/text_to_image/train_text_to_image_lora_sdxl.py

先用 --stop-after 8 检查一次数据遍历，再从step8恢复到配置中的80步。
stop-after不改配置，所以恢复时能严格核对同一个实验。
"""
from common import ROOT, LAB, read, write, sha, local_environment, config, checkpoint
import argparse
import datetime
import json
import random
import subprocess
import threading
import time


def example_at(micro_index, count, seed):
    """每轮洗牌一次，无放回遍历；位置由步数决定，恢复不会重头读前几张。"""
    epoch, offset=divmod(micro_index,count)
    order=list(range(count))
    random.Random(seed+epoch).shuffle(order)
    return order[offset]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stop-after',type=int)
    parser.add_argument('--resume',type=str)
    parser.add_argument('--run-name',default='full')
    a=parser.parse_args()
    assert a.run_name.replace('-','').replace('_','').isalnum()
    cfg=config()
    assert cfg['batch_size']==1 and cfg['mixed_precision']=='fp16'
    assert cfg['optimizer']=='AdamW8bit' and cfg['gradient_checkpointing']
    assert sha(ROOT/'data/manifest.json')==cfg['manifest_sha256']
    cache_index=read(LAB/'cache/index.json')
    assert cache_index['status']=='success'
    assert cache_index['identity']['manifest_sha256']==cfg['manifest_sha256']
    assert cache_index['identity']['base_sha256']==cfg['base_sha256']
    manifest=read(ROOT/'data/manifest.json')
    training=[r for r in manifest['examples'] if r['split']=='train']
    assert [r['id'] for r in training]==[r['id'] for r in cache_index['entries']]
    for row,entry in zip(training,cache_index['entries']):
        assert sha(ROOT/entry['path'])==entry['sha256']
        assert sha(ROOT/row['image'])==row['image_sha256']
        assert sha(ROOT/row['caption_path'])==row['caption_sha256']
    log=LAB/'logs'/f'train-{a.run_name}'
    summary_path=log.with_suffix('.summary.json')
    if summary_path.exists():
        old=read(summary_path)
        if old['status'] in ('success','stopped_at_requested_step'):
            assert old['config']==cfg
            assert all(sha(ROOT/x['path'])==x['sha256'] for x in old['checkpoints'])
            print('SKIP completed process:',a.run_name); return
        raise RuntimeError('Previous failure retained; choose an explicit new run-name and resume state')
    # 不终止用户程序。显存不足时在加载模型前停止，保留清楚的原因。
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                                     '--format=csv,noheader,nounits','--id=0'],text=True).strip())
    if free<6144:
        raise RuntimeError(f'Only {free} MiB free VRAM; close other GPU workloads, then rerun. No training started.')
    import torch
    import torch.nn.functional as F
    import bitsandbytes as bnb
    from diffusers import UNet2DConditionModel, DDPMScheduler
    from peft import LoraConfig, set_peft_model_state_dict
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file,save_file
    from diffusers.utils.state_dict_utils import convert_state_dict_to_kohya
    torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    # 对已知非确定性算子警告；不承诺跨GPU逐像素复现。
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter(); samples=[]; stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                v=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used',
                                           '--format=csv,noheader,nounits','--id=0'],text=True,timeout=5)
                samples.append({'elapsed_seconds':time.perf_counter()-start,'whole_gpu_mib':int(v.strip())})
            except Exception: pass
            stop.wait(0.5)
    thread=threading.Thread(target=monitor,daemon=True); thread.start()
    summary={'status':'running','date':datetime.datetime.now().astimezone().isoformat(),
             'config':cfg,'cache_fingerprint':cache_index['fingerprint'],
             'trainer_sha256':sha(__file__),'environment':local_environment(),
             'run_name':a.run_name,'checkpoints':[],'records':[],
             'resumed_from':a.resume, 'timing_scope':'model load + train + checkpoint save; excludes separate cache process'}
    first=0
    try:
        # 1) 冻结基础UNet；只向注意力投影添加小型LoRA矩阵。
        unet=UNet2DConditionModel.from_single_file(str(checkpoint()),
             config=str(ROOT/'configs/base'),subfolder='unet',
             local_files_only=True,torch_dtype=torch.float16)
        unet.requires_grad_(False)
        unet.add_adapter(LoraConfig(r=cfg['rank'],lora_alpha=cfg['lora_alpha'],
                                   init_lora_weights='gaussian',target_modules=cfg['target_modules']))
        parameters=[]
        for name,value in unet.named_parameters():
            if value.requires_grad:
                assert 'lora_' in name, f'Unexpected full-model trainable parameter: {name}'
                value.data=value.data.float() # 小矩阵用FP32维护；大模型仍为FP16。
                parameters.append(value)
        unet.enable_gradient_checkpointing()
        unet.to('cuda'); unet.train()
        summary['trainable_parameters']=sum(p.numel() for p in parameters)
        summary['total_parameters']=sum(p.numel() for p in unet.parameters())
        optimizer=bnb.optim.AdamW8bit(parameters,lr=cfg['learning_rate'],betas=(0.9,0.999),
                                     weight_decay=cfg['weight_decay'],eps=1e-8)
        scaler=torch.amp.GradScaler('cuda')
        noise_scheduler=DDPMScheduler.from_pretrained(str(ROOT/'configs/base'),
                                                     subfolder='scheduler',local_files_only=True)
        assert noise_scheduler.config.prediction_type=='epsilon'
        # 2) 小数据集缓存放CPU内存；每个microbatch只把一张的张量送进GPU。
        caches=[load_file(str(ROOT/r['path'])) for r in cache_index['entries']]
        if a.resume:
            resume=(ROOT/a.resume).resolve()
            assert resume.is_relative_to((LAB/'checkpoints').resolve()), 'Use only this project trusted local states'
            state=torch.load(resume,map_location='cpu',weights_only=False)
            assert state['config']==cfg and state['cache_fingerprint']==cache_index['fingerprint']
            assert state['trainer_sha256']==sha(__file__), 'Trainer changed since saved state'
            summary['resumed_trainer_sha256']=state['trainer_sha256']
            set_peft_model_state_dict(unet,state['adapter'])
            optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
            first=state['step']; summary['records']=state['records']
            torch.set_rng_state(state['cpu_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
            summary['restore_audit']={'step':first,'optimizer_state_entries':len(optimizer.state),
                                      'adapter_tensors':len(state['adapter']),
                                      'optimizer_loaded':True,'scaler_loaded':True,'cpu_cuda_rng_loaded':True,
                                      'next_micro_index':first*cfg['gradient_accumulation']}
            del state
        end=a.stop_after if a.stop_after is not None else cfg['max_steps']
        assert first<end<=cfg['max_steps']
        optimizer.zero_grad(set_to_none=True)
        # 3) 每4张分别反向传播一次，累积后更新1次LoRA。step指优化器更新，不是图片数。
        for step in range(first+1,end+1):
            tick=time.perf_counter(); losses=[]; used=[]
            for micro in range(cfg['gradient_accumulation']):
                pos=(step-1)*cfg['gradient_accumulation']+micro
                idx=example_at(pos,len(caches),cfg['seed']); used.append(training[idx]['id'])
                cache={k:v.to('cuda',dtype=torch.float16) for k,v in caches[idx].items()}
                latent=cache['latents']
                noise=torch.randn_like(latent)
                timestep=torch.randint(0,noise_scheduler.config.num_train_timesteps,(1,),device='cuda').long()
                noisy=noise_scheduler.add_noise(latent,noise,timestep)
                with torch.autocast('cuda',dtype=torch.float16):
                    pred=unet(noisy,timestep,cache['prompt_embeds'],
                              added_cond_kwargs={'text_embeds':cache['pooled_prompt_embeds'],
                                                 'time_ids':cache['time_ids']},return_dict=False)[0]
                    loss=F.mse_loss(pred.float(),noise.float())
                assert torch.isfinite(loss), 'Nonfinite loss; do not silently retry'
                losses.append(loss.item())
                scaler.scale(loss/cfg['gradient_accumulation']).backward()
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['max_grad_norm'])
            assert torch.isfinite(norm), 'Nonfinite gradients; update not accepted'
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            record={'step':step,'sample_ids':used,'loss':sum(losses)/len(losses),
                    'gradient_norm':float(norm),'step_seconds':time.perf_counter()-tick,
                    'elapsed_seconds':time.perf_counter()-start,
                    'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated()}
            summary['records'].append(record)
            with log.with_suffix('.loss.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            # 4) 同时保存“用于出图的小权重”和“用于恢复的完整训练状态”。
            if step in cfg['checkpoint_steps'] or step==end:
                adapter={k:v.detach().cpu().contiguous() for k,v in get_peft_model_state_dict(unet).items()}
                kohya=convert_state_dict_to_kohya({'unet.'+k:v for k,v in adapter.items()})
                dst=LAB/'checkpoints'/f'{cfg["experiment_id"]}_seed{cfg["seed"]}_step{step:04d}.safetensors'
                assert not dst.exists(), 'Do not overwrite an existing checkpoint'
                temp=dst.with_suffix('.tmp')
                save_file(kohya,str(temp),metadata={'ss_base_model_version':'sdxl_base_v1-0',
                          'rank':str(cfg['rank']),'steps':str(step),'dataset_manifest_sha256':cfg['manifest_sha256'],
                          'purpose':'CC0 Kenney icon style demo; no quality guarantee'})
                temp.replace(dst)
                summary['checkpoints'].append({'path':str(dst.relative_to(ROOT)),'sha256':sha(dst),
                                                'step':step,'bytes':dst.stat().st_size})
                resume_dst=dst.with_suffix('.resume.pt'); resume_temp=resume_dst.with_suffix('.tmp')
                torch.save({'adapter':adapter,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                            'step':step,'records':summary['records'],'config':cfg,
                            'cache_fingerprint':cache_index['fingerprint'],'trainer_sha256':sha(__file__),
                            'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},resume_temp)
                resume_temp.replace(resume_dst)
        summary['status']='success' if end==cfg['max_steps'] else 'stopped_at_requested_step'
        summary['last_step']=end
        summary['steps_executed_this_process']=end-first
        summary['unique_training_images_seen']=len({i for r in summary['records'] for i in r['sample_ids']})
    except BaseException as e:
        summary.update(status='failed',error=repr(e)); raise
    finally:
        stop.set(); thread.join(timeout=6)
        summary.update(wall_seconds=time.perf_counter()-start,
                       peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
                       whole_gpu_peak_sampled_mib=max((r['whole_gpu_mib'] for r in samples),default=None))
        write(summary_path,summary); write(log.with_suffix('.gpu.json'),samples)
        print(json.dumps({'status':summary['status'],'seconds':summary['wall_seconds'],
                          'last_step':summary.get('last_step')}),flush=True)


if __name__=='__main__': main()
