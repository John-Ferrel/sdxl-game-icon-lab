"""只导出本仓库公开数据实验的图片与摘要；不复制原始日志、路径、缓存或权重。"""
from common import ROOT, LAB, read, write, sha, config
from PIL import Image, ImageDraw
from safetensors import safe_open
from pathlib import Path
import csv
import json
import shutil


def contact(rows,path,arms):
    groups=[]
    for subject in ['compass','backpack']:
        for seed in [20261026,20261027]:
            group=[next((r for r in rows if r['id']==f'PUBLIC-{subject}-{arm}' and r['seed']==seed),None) for arm in arms]
            if all(group):groups.append(group)
    out=Image.new('RGB',(len(arms)*280,len(groups)*310),'white');d=ImageDraw.Draw(out)
    for y,group in enumerate(groups):
        for x,row in enumerate(group):
            im=Image.open(ROOT/row['image']);im.thumbnail((270,270))
            d.text((x*280+5,y*310+4),row['id'].replace('PUBLIC-','')+' / '+str(row['seed']),fill='black')
            out.paste(im,(x*280+5,y*310+28))
    out.save(path)


def main():
    cfg=config();manifest=read(ROOT/'data/manifest.json')
    assert manifest['license']=='CC0-1.0' and cfg['experiment_id']=='PUBLIC-ICON-R8'
    # 恢复进程可以换run-name；选择真正到达80步的成功汇总，而非猜文件名。
    finished=[read(p) for p in (LAB/'logs').glob('train-*.summary.json')]
    finals=[r for r in finished if r.get('status')=='success' and r.get('last_step')==80]
    assert len(finals)==1, 'Need one completed80-step run before publishing results'
    summary=finals[0]
    smoke=read(LAB/'logs/train-smoke.summary.json')
    assert summary['status']=='success' and summary['last_step']==80
    interruption=read(LAB/'logs/interrupted-run.json') if (LAB/'logs/interrupted-run.json').exists() else None
    completed_seconds=sum(r['wall_seconds'] for r in finished if r.get('status') in ('success','stopped_at_requested_step'))
    interrupted_seconds=interruption['recorded_elapsed_lower_bound_seconds'] if interruption else 0
    executed_updates=sum(r['steps_executed_this_process'] for r in finished if r.get('status') in ('success','stopped_at_requested_step'))
    if interruption: executed_updates+=interruption['recorded_process_updates']
    records=[]
    for p in sorted((LAB/'logs').glob('PUBLIC-*.json')):
        r=read(p)
        if not isinstance(r,dict) or 'images' not in r or 'status' not in r:continue
        assert r['status']=='success'
        if r['lora']:
            weight=LAB/'checkpoints'/r['lora']['name']
            assert weight.name.startswith('PUBLIC-ICON-R8_') and sha(weight)==r['lora']['sha256']
            with safe_open(str(weight),framework='pt',device='cpu') as f:
                assert f.metadata()['dataset_manifest_sha256']==cfg['manifest_sha256']
        assert len(r['images'])==1
        raw=ROOT/r['images'][0]['path']; assert sha(raw)==r['images'][0]['sha256']
        dest=ROOT/'results/images'/raw.name;dest.parent.mkdir(parents=True,exist_ok=True)
        # 只去掉PNG运行元数据，RGB像素保持不变；参数另存干净的JSONL。
        with Image.open(raw) as im:
            clean=Image.new('RGB',im.size);clean.paste(im.convert('RGB'));clean.save(dest)
        with Image.open(dest) as im:
            import hashlib
            assert hashlib.sha256(im.tobytes()).hexdigest()==r['images'][0]['rgb_sha256']
        fields=['id','seed','resolution','steps','cfg','sampler','scheduler','denoise','positive','negative',
                'base_model','controlnet','lora','wall_seconds','peak_torch_allocated_bytes','gpu_total_peak_sampled_mib']
        records.append({**{k:r[k] for k in fields},'image':dest.relative_to(ROOT).as_posix(),
                        'image_sha256':sha(dest),'rgb_sha256':r['images'][0]['rgb_sha256']})
    paired=[r for r in records if r['resolution']==[512,512]]
    assert len(paired)==12,'Do not export a cherry-picked partial comparison'
    with (ROOT/'results/experiments.jsonl').open('w',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    contact(paired,ROOT/'results/all-comparisons.png',['base','40','80'])
    contact([r for r in paired if r['seed']==20261026],ROOT/'docs/images/public-comparison.png',['base','40','80'])
    with (ROOT/'results/training-loss.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['step','loss','sample_ids'])
        for r in summary['records']:w.writerow([r['step'],r['loss'],' '.join(r['sample_ids'])])
    inference=read(LAB/'logs/PUBLIC-compass-base_seed20261026.environment.json')
    state={'dataset':'Kenney Generic Items / CC0','training_images':32,'holdout_images':8,
           'dataset_manifest_sha256':cfg['manifest_sha256'],'optimizer_steps':80,
           'sample_uses':320,'passes_per_image':10,'checkpoint_steps':[8,40,80],
           'training_completed_process_seconds':completed_seconds,
           'training_recorded_compute_lower_bound_seconds':completed_seconds+interrupted_seconds,
           'recorded_executed_updates_including_replayed_work':executed_updates,
           'interruption':interruption,
           'training_peak_torch_allocated_gib':max(smoke['peak_torch_allocated_bytes'],summary['peak_torch_allocated_bytes'])/2**30,
           'training_peak_whole_gpu_sampled_mib':max(smoke['whole_gpu_peak_sampled_mib'],summary['whole_gpu_peak_sampled_mib']),
           'paired_images':len(paired),'additional_resolution_tests':len(records)-len(paired),
           'paired_inference_node_seconds_range':[min(r['wall_seconds'] for r in paired),max(r['wall_seconds'] for r in paired)],
           'gpu':inference['gpu'],'torch':inference['torch'],'torch_cuda':inference['torch_cuda'],
           'packages':summary['environment']['packages'],'configuration':cfg,
           'source_sha256':{p.name:sha(p) for p in (ROOT/'scripts').glob('*.py')},
           'limits':['non-blind small-sample visual observations; no human ratings collected',
                     'paired prompt subject categories occur in the training set; no strict unseen-category claim',
                     'no data-size/epoch/learning-rate ablation; no statistical significance claim',
                     'existing environment reused; clean-machine install not separately tested',
                     'attention backward may be nondeterministic; no cross-GPU bitwise promise']}
    write(ROOT/'results/summary.json',state)
    write(ROOT/'results/dataset-manifest.json',manifest)
    lines=['# 公开素材实验：实际记录','',
           '这里仅包含Kenney公开素材独立实验；图片导出时移除运行元数据，RGB像素未改变。参数单独保存在JSONL。','',
           f'- 数据：32训练、8留出；80次更新，320次图片使用，每图10次。',
           f'- 已正常结束的训练进程合计{state["training_completed_process_seconds"]:.2f}秒；计入已记录中断段后，计算时间至少{state["training_recorded_compute_lower_bound_seconds"]:.2f}秒。中断退出耗时未知，不把缺失时间计为零。',
           f'- 已正常结束进程的torch allocated峰值{state["training_peak_torch_allocated_gib"]:.3f}GiB；整卡采样峰值{state["training_peak_whole_gpu_sampled_mib"]}MiB。两者口径不同，中断段的完整显存采样未保存。',
           f'- 对照：12张512²；节点执行约{state["paired_inference_node_seconds_range"][0]:.2f}–{state["paired_inference_node_seconds_range"][1]:.2f}秒/张，排除进程启动/import/验证。',
           f'- 额外尺寸检查：{state["additional_resolution_tests"]}张，单独记录，不混入配对质量比较。','',
           '## 所有对照，不只选最好看的','',
           '每行相同Prompt和seed，三列依次Base、40步、80步。','',
           '![两条Prompt、两个seed的全部对照](all-comparisons.png)','',
           '| 实验 | 尺寸 | 节点秒 | 图片 |','|---|---|---:|---|']
    for r in records:
        lines.append(f'| {r["id"]} / {r["seed"]} | {r["resolution"][0]}×{r["resolution"][1]} | {r["wall_seconds"]:.2f} | [PNG]({Path(r["image"]).relative_to("results").as_posix()}) |')
    wides=[r for r in records if r['resolution']!=[512,512]]
    if wides:
        lines+=['','## 尺寸检查（单独条件）','',
                '768×512成功保存，说明本例没有被训练尺寸锁死；画面仍有发虚与伪文字。更大尺寸不自动等于更清晰。','']
        for r in wides:
            lines += [f'![独立尺寸检查：{r["resolution"][0]}×{r["resolution"][1]}]({Path(r["image"]).relative_to("results").as_posix()})','']
    lines+=['## 初步目视观察（AI助手辅助，待用户复核）','',
            '- seed20261026指南针：80步更简化，但黄铜与蓝色玻璃中心等要求弱化。',
            '- 背包：后期趋向扁平图标，前袋等细节可能丢失；seed20261027仍出现多视图。',
            '- 这些现象说明风格发生变化，不构成总体质量或泛化改善的证明。','']
    lines+=['','## 解释边界','',
            f'最终权重是80步，但本机实际记录了{executed_updates}次更新（包括重放）。中断记录见摘要中的interruption；本次从40恢复，41–70步重新计算，最终训练轨迹仍是1–80。',
            '没有用户打分，不给出“画质提升百分比”。画风变化、Prompt遵循、几何合理性和多样性需要分别检查。数据量/训练时间不足只是待验证解释，不是本轮证实的根因。',
            '基础模型、LoRA权重、恢复文件、缓存及原始机器日志未加入公开仓库。本地运行记录在Git忽略的runs/目录。','',
            '[完整参数与性能JSONL](experiments.jsonl) / [摘要](summary.json) / [训练loss CSV](training-loss.csv) / [数据指纹](dataset-manifest.json)。','']
    (ROOT/'results/README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'published_paired_images':len(paired),'resolution_tests':len(records)-len(paired),
                      'training_recorded_lower_bound_seconds':state['training_recorded_compute_lower_bound_seconds']}))


if __name__=='__main__':main()
