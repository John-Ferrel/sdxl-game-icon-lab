"""第1步，仅CPU：40张公开素材 → 32训练/8留出，透明图合成灰底512方图。"""
from common import ROOT, LAB, write, sha
from PIL import Image, ImageDraw
import csv


def main():
    # 不自动下载大模型。40张原PNG和描述已经随仓库提供。
    with (ROOT/'data/captions.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==40
    examples=[]
    for row in rows:
        src=ROOT/'data/examples'/row['file']
        with Image.open(src) as original:
            im=original.convert('RGBA')
            box=im.getchannel('A').getbbox()
            if box is None: raise ValueError(f'Empty icon: {src}')
            im=im.crop(box)
            # 保留比例；最长边384，留出边距。上采样不等于增加真实细节。
            scale=384/max(im.size)
            im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS)
            canvas=Image.new('RGB',(512,512),'#f2f2f2')
            canvas.paste(im,((512-im.width)//2,(512-im.height)//2),im)
        dst=ROOT/'data/prepared'/row['split']/(row['id']+'.png')
        dst.parent.mkdir(parents=True,exist_ok=True)
        # 缓存开始后拒绝改变这版数据；新尝试应使用新的run_dir并重新建立缓存。
        if (LAB/'cache/index.json').exists():
            if not dst.exists(): raise RuntimeError('Prepared dataset changed after caching')
        else:
            canvas.save(dst)
            dst.with_suffix('.txt').write_text(row['caption']+'\n',encoding='utf-8')
        assert dst.with_suffix('.txt').read_text(encoding='utf-8').strip()==row['caption']
        examples.append({'id':row['id'],'split':row['split'],'caption':row['caption'],
                         'image':dst.relative_to(ROOT).as_posix(),
                         'caption_path':dst.with_suffix('.txt').relative_to(ROOT).as_posix(),
                         'image_sha256':sha(dst),'caption_sha256':sha(dst.with_suffix('.txt')),
                         'source':src.relative_to(ROOT).as_posix(),'source_sha256':sha(src)})
    train=[r for r in examples if r['split']=='train']
    holdout=[r for r in examples if r['split']=='holdout']
    assert len(train)==32 and len(holdout)==8
    assert len({r['source_sha256'] for r in examples})==40
    write(ROOT/'data/manifest.json',{'source':'data/SOURCE.json','license':'CC0-1.0',
          'preprocessing':'alpha crop; preserve aspect; longest side384; center on512 RGB #f2f2f2',
          'train_count':32,'holdout_count':8,'examples':examples})
    for name,group in [('dataset-train',train),('dataset-holdout',holdout)]:
        out=Image.new('RGB',(960,((len(group)+7)//8)*145),'white');draw=ImageDraw.Draw(out)
        for i,row in enumerate(group):
            im=Image.open(ROOT/row['image']);im.thumbnail((116,116))
            x=(i%8)*120;y=(i//8)*145
            out.paste(im,(x,y));draw.text((x+4,y+120),row['id'],fill='black')
        out.save(ROOT/'docs/images'/f'{name}.png')
    for folder in ['logs','cache','checkpoints','outputs','workflows']:(LAB/folder).mkdir(parents=True,exist_ok=True)
    print('Prepared: 32 train / 8 holdout. No model loaded; original CC0 PNGs unchanged.')


if __name__=='__main__': main()
