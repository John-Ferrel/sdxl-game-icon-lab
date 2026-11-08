"""检查本项目可公开文件；可选生成小型ZIP，不接触远程GitHub。"""
from common import ROOT, read, sha, write
from pathlib import Path
import argparse
import ast
import json
import os
import re
import zipfile

EXCLUDED={'runs','tools','.venv','.git','dist','__pycache__'}
ROOT_FILES={'README.md','LICENSE','NOTICE','.gitignore','config.example.json','requirements.txt'}


def public_files():
    for current,dirs,names in os.walk(ROOT):
        relative=Path(current).relative_to(ROOT)
        dirs[:]=[d for d in dirs if d not in EXCLUDED
                 and not (relative==Path('data') and d in {'source','prepared'})]
        for name in names:
            p=Path(current)/name;r=p.relative_to(ROOT)
            if len(r.parts)==1 and name not in ROOT_FILES:continue
            if r.as_posix()=='data/manifest.json':continue
            if p.suffix in {'.pyc','.pt','.safetensors','.ckpt','.zip'}:continue
            yield p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',action='store_true')
    args=parser.parse_args();files=sorted(public_files());names={p.relative_to(ROOT).as_posix() for p in files}
    required={'README.md','LICENSE','NOTICE','data/captions.csv','licenses/KENNEY-CC0.txt',
              'results/summary.json','results/experiments.jsonl','results/README.md',
              'docs/blog.zh-CN.md','docs/SETUP.md','workflows/01_base.ui.json','workflows/02_lora.ui.json'}
    assert required<=names, f'Missing published files: {required-names}'
    assert not any('config.local.json' in n or n.startswith(('runs/','data/source/')) for n in names)
    # 当前电脑的项目父目录/用户目录不应出现在公开文字里；示例路径可以保留。
    private_paths=[str(ROOT.parent),ROOT.parent.as_posix(),str(Path.home()),Path.home().as_posix()]
    missing=[]
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'),filename=p.name)
        if p.suffix not in {'.md','.py','.json','.jsonl','.txt','.csv','.ui','.gitignore'} and p.name not in ROOT_FILES:continue
        text=p.read_text(encoding='utf-8-sig')
        assert '\x00' not in text, f'NUL in {p.name}'
        # 同时检查JSON编码过的Windows路径。
        for private in private_paths:
            assert private not in text and json.dumps(private)[1:-1] not in text, f'Local path in {p.relative_to(ROOT)}'
        if p.suffix=='.md':
            text=re.sub(r'```[\s\S]*?```','',text)
            text=re.sub(r'`[^`\n]*`','',text)
            for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
                if link.startswith(('https:','http:','#')):continue
                target=(p.parent/link.split('#',1)[0]).resolve()
                if not target.is_relative_to(ROOT.resolve()) or not target.exists():missing.append([p.name,link])
    assert not missing,missing
    manifest=read(ROOT/'results/dataset-manifest.json')
    assert manifest['license']=='CC0-1.0'
    for row in manifest['examples']:
        assert row['source'] in names and sha(ROOT/row['source'])==row['source_sha256']
    for line in (ROOT/'results/experiments.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        assert row['image'] in names and sha(ROOT/row['image'])==row['image_sha256']
    summary=read(ROOT/'results/summary.json')
    assert summary['paired_images']==12 and summary['optimizer_steps']==80
    report={'public_files':len(files),'total_bytes':sum(p.stat().st_size for p in files),
            'private_paths_found':False,'model_weights_included':False,'dataset_license':'CC0-1.0',
            'all_local_markdown_links_exist':True,'all_public_image_hashes_match':True}
    if args.package:
        dst=ROOT/'dist/sdxl-game-icon-lab-public.zip';dst.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(dst,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,(Path(ROOT.name)/p.relative_to(ROOT)).as_posix())
        report.update(package=dst.relative_to(ROOT).as_posix(),package_bytes=dst.stat().st_size,
                      package_sha256=sha(dst))
    write(ROOT/'dist/publication-audit.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
