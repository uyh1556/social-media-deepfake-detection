# Colab: S1~S6 seed 43·44 추가 학습

권장 노트북 이름은 `train_s1_fixed_q95_seed43_44.ipynb` 또는
`train_s1_mixed_jpeg_seed43_44.ipynb` 형식이다. 한 노트북은 하나의 selection과
하나의 protocol만 담당한다. 따라서 전체를 병렬 운영하려면 최대 12개 노트북을
만들되, 실제 동시 실행 수는 Colab 제한에 맞춘다.

## Cell 1 — Drive, GPU, 코드

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, torch

assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))

PROJECT=Path('/content/project')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/social-media-deepfake-detection.git',str(PROJECT)],check=True)
subprocess.run(['pip','install','-q','-r',str(PROJECT/'requirements-colab.txt')],check=True)
```

## Cell 2 — 담당 조건 설정

```python
SELECTION='S1'       # S1, S2, S3, S4, S5, S6
PROTOCOL='fixed_q95' # fixed_q95 또는 mixed_jpeg

ROOT=Path('/content/drive/MyDrive/deepfake_thesis')
DRIVE_DATA=ROOT/'data/family_rotation_v1'
DATA_ROOT=Path('/content/deepfake_family_rotation_v1')
MANIFEST_ROOT=Path('/content/family_rotation_manifests')
RUNS_ROOT=ROOT/'runs/family_rotation_v1'
```

## Cell 3 — 필요한 Real 및 6개 method TAR만 압축 해제

```python
import json, shutil, sys
sys.path.insert(0,'/content/project/scripts')
from create_family_rotation_method_archives import METHOD_SLUGS

config=json.loads((PROJECT/'configs/family_rotation_v1/selections.json').read_text())
letters=config['selections'][SELECTION]
methods=[config['families'][family][letter] for family in ('FS','FR','EFS') for letter in letters]
archives=[DRIVE_DATA/'real_trainval_v1.tar']+[
    DRIVE_DATA/f'df40_{METHOD_SLUGS[method]}_trainval_v1.tar' for method in methods
]
for archive in archives:
    assert archive.is_file(), archive
    marker=Path('/content')/f'.seed_repeats_{archive.name}.done'
    if marker.exists():
        print('Already extracted:',archive.name,flush=True)
        continue

    # Drive 마운트에서 tar를 직접 읽으면 간헐적으로 exit status 2가 발생한다.
    # 한 개씩 Colab 로컬 디스크로 복사한 뒤 풀고 임시 복사본을 제거한다.
    local_archive=Path('/content')/archive.name
    print('Copying to local disk:',archive.name,flush=True)
    if local_archive.exists():
        local_archive.unlink()
    shutil.copyfile(archive,local_archive)
    assert local_archive.stat().st_size == archive.stat().st_size, archive
    try:
        print('Extracting locally:',archive.name,flush=True)
        result=subprocess.run(
            ['tar','-xf',str(local_archive),'-C','/content'],
            text=True,capture_output=True,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError(f'tar failed: {archive.name}')
        marker.touch()
    finally:
        local_archive.unlink(missing_ok=True)
print(SELECTION,methods)
```

## Cell 4 — 동일한 frozen manifest 생성

```python
subprocess.run([
    'python','-u','/content/project/scripts/create_family_rotation_conditions.py',
    '--module-root',str(DATA_ROOT),'--output-dir',str(MANIFEST_ROOT),
    '--selections',SELECTION,
],check=True)
```

## Cell 5 — seed 43, 44 순차 학습

```python
!python -u /content/project/scripts/train_xception_family_rotation_repeated_seeds.py \
  --selection "{SELECTION}" --protocol "{PROTOCOL}" \
  --seeds 43 44 \
  --data-root "{DATA_ROOT}" --manifest-root "{MANIFEST_ROOT}" \
  --output-root "{RUNS_ROOT}" \
  --epochs 15 --batch-size 16 --workers 2 \
  --learning-rate 1e-4 --weight-decay 1e-4 --patience 3
```

진행률은 각 M1~M7의 `Train:`과 `Validation:`으로 바로 출력된다. 중단되면 Cell 5를
다시 실행한다. 완료된 checkpoint는 건너뛰고 미완료 checkpoint는 `last.pt`에서
재개하므로 기존 seed 42 및 완료된 seed 43·44 결과를 덮어쓰지 않는다.
