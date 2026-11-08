"""独立进程执行原生ComfyUI节点，保存参数/PNG/耗时/显存。不会自动下载模型。"""
import sys
sys.dont_write_bytecode = True
import os
from pathlib import Path
from common import ROOT, LAB, config, checkpoint, comfyui, settings
os.environ.update(PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', HF_HOME=str(ROOT/'runs/hf-cache'), TORCH_HOME=str(ROOT/'runs/torch-cache'))
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import threading
import time


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def ui_graph(api, mappings):
    nodes, links = [], []
    ids = {key: i+1 for i,key in enumerate(api)}
    output_lists = {}
    for order,(key,spec) in enumerate(api.items()):
        cls=mappings[spec['class_type']]
        schema=cls.INPUT_TYPES()
        inputs,widgets=[],[]
        names=list(schema.get('required',{}))+list(schema.get('optional',{}))
        for name in names:
            if name not in spec['inputs']: continue
            value=spec['inputs'][name]
            if isinstance(value,list):
                source,slot=value; typ=mappings[api[source]['class_type']].RETURN_TYPES[slot]
                lid=len(links)+1; links.append([lid,ids[source],slot,ids[key],len(inputs),str(typ)])
                inputs.append({'name':name,'type':str(typ),'link':lid})
            else:
                widgets.append(value)
                if name in ('seed','noise_seed'): widgets.append('fixed')
        types=cls.RETURN_TYPES; labels=getattr(cls,'RETURN_NAMES',types)
        outs=[{'name':str(label),'type':str(typ),'links':[]} for label,typ in zip(labels,types)]
        output_lists[ids[key]]=outs
        nodes.append({'id':ids[key],'type':spec['class_type'],'pos':[order%4*390,order//4*330],'size':[360,290], 'flags':{},'order':order,'mode':0,'inputs':inputs,'outputs':outs,'properties':{'Node name for S&R':spec['class_type']},'widgets_values':widgets})
    for lid,src,slot,*_ in links: output_lists[src][slot]['links'].append(lid)
    return {'last_node_id':len(nodes),'last_link_id':len(links),'nodes':nodes,'links':links,'groups':[],'config':{},'extra':{},'version':0.4}


def execute(job_path):
    job=json.loads(Path(job_path).read_text(encoding='utf8')); stage=ROOT/job['stage']
    key=f"{job['id']}_seed{job['seed']}"; result_path=stage/'logs'/f'{key}.json'
    fingerprint=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
    if result_path.exists():
        old=json.loads(result_path.read_text())
        if old.get('status')=='success':
            if old.get('job_sha256')!=fingerprint: raise RuntimeError('ID conflict: completed experiment has different parameters')
            if all(digest(ROOT/x['path'])==x['sha256'] for x in old['images']):
                print('SKIP completed',key);return
        raise RuntimeError('Prior incomplete/failed experiment retained; use a new explicit experiment ID')
    cfg=config()
    sys.path.insert(0,str(comfyui()))
    sys.argv=[sys.argv[0]]+cfg['runtime_args']
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    assert torch.__version__=='2.6.0+cu124' and torch.cuda.is_available()
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import folder_paths,nodes,execution
    from comfy_extras.nodes_canny import Canny
    nodes.NODE_CLASS_MAPPINGS['Canny']=Canny
    folder_paths.set_output_directory(str(stage/'outputs'));folder_paths.set_input_directory(str(ROOT/'data/input'))
    folder_paths.add_model_folder_path('checkpoints',str(checkpoint().parent))
    # ControlNet仅供选读示例，不自动下载。
    if settings().get('controlnet'):
        folder_paths.add_model_folder_path('controlnet',str(Path(settings()['controlnet']).resolve().parent))
    folder_paths.add_model_folder_path('loras',str(LAB/'checkpoints'))
    api=job['workflow']; ui=ui_graph(api,nodes.NODE_CLASS_MAPPINGS)
    write(stage/'workflows'/f'{key}.api.json',api);write(stage/'workflows'/f'{key}.ui.json',ui)
    validation=execution.validate_prompt(api)
    write(stage/'logs'/f'{key}.validation.json',validation)
    if not validation[0]: raise RuntimeError(f'Invalid workflow: {validation}')
    env={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'runtime_args':cfg['runtime_args'],'comfyui_commit':cfg['comfyui_commit'],'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'runner_sha256':digest(__file__),'tf32':False}
    write(stage/'logs'/f'{key}.environment.json',env)
    result={**{k:v for k,v in job.items() if k!='workflow'},'job_sha256':fingerprint,'workflow_sha256':hashlib.sha256(json.dumps(api,sort_keys=True).encode()).hexdigest(),'workflow_path':str((stage/'workflows'/f'{key}.api.json').relative_to(ROOT)),'date':dt.datetime.now().astimezone().isoformat(),'environment_path':str((stage/'logs'/f'{key}.environment.json').relative_to(ROOT)),'status':'running','images':[],'timing_scope':'native node execution including load/text/sampling/VAE/save; excludes process/import and validation','node_seconds':{}}
    samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5)
                samples.append({'elapsed':time.perf_counter()-start,'used_mib':float(raw.strip())})
            except Exception: pass
            stop.wait(.5)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        values={}
        with torch.inference_mode():
            for node,spec in api.items():
                obj=nodes.NODE_CLASS_MAPPINGS[spec['class_type']]()
                kwargs={name:values[v[0]][v[1]] if isinstance(v,list) else v for name,v in spec['inputs'].items()}
                if spec['class_type']=='SaveImage':kwargs.update(prompt=api,extra_pnginfo={'workflow':ui,'experiment':{k:v for k,v in job.items() if k!='workflow'}})
                tick=time.perf_counter(); out=getattr(obj,obj.FUNCTION)(**kwargs);torch.cuda.synchronize()
                result['node_seconds'][node+':'+spec['class_type']]=time.perf_counter()-tick
                if isinstance(out,dict):
                    for im in out.get('ui',{}).get('images',[]):
                        path=stage/'outputs'/im['subfolder']/im['filename']
                        from PIL import Image
                        with Image.open(path) as p: rgb=p.convert('RGB');ph=hashlib.sha256(rgb.tobytes()).hexdigest();dims=list(rgb.size)
                        result['images'].append({'path':str(path.relative_to(ROOT)),'sha256':digest(path),'rgb_sha256':ph,'dimensions':dims})
                    values[node]=out.get('result',())
                else:values[node]=out
        assert result['images'];result['status']='success'
    except Exception as e:
        result.update(status='failed',error=repr(e));raise
    finally:
        result['wall_seconds']=time.perf_counter()-start
        result['peak_torch_allocated_bytes']=torch.cuda.max_memory_allocated();result['peak_torch_reserved_bytes']=torch.cuda.max_memory_reserved()
        stop.set();thread.join(timeout=6)
        result['gpu_total_peak_sampled_mib']=max((x['used_mib'] for x in samples),default=None)
        result['vram_limitations']='whole GPU sampling includes desktop, nominal 500ms, can miss transient peaks; allocator metrics exclude non-torch allocations'
        write(result_path,result);write(stage/'logs'/f'{key}.gpu.json',samples)
    print(json.dumps({'id':key,'status':result['status'],'seconds':result['wall_seconds'],'torch_peak_gib':result['peak_torch_allocated_bytes']/2**30,'images':result['images']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');execute(p.parse_args().job)
