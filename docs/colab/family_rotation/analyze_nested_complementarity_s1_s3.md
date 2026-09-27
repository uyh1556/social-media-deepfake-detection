# Colab: S1-S3 nested complementarity 검증

권장 노트북 이름: `analyze_nested_complementarity_s1_s3.ipynb`

새 모델을 학습하지 않는다. 각 selection에서 한 method를 완전히 숨긴 뒤, 남은 5개
single-method expert를 균일하게 합친 결과와 전이행렬 상보성으로 합친 결과를 비교한다.

## Cell 1 — 준비

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess

PROJECT=Path('/content/project')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/social-media-deepfake-detection.git',str(PROJECT)],check=True)
subprocess.run(['pip','install','-q','pandas','numpy','scipy','tabulate'],check=True)
ROOT=Path('/content/drive/MyDrive/deepfake_thesis')
```

## Cell 2 — S1·S2·S3 분석

```python
TRANSFER_ROOT=ROOT/'evaluations/family_rotation_method_transfer_v1'
OUTPUT_DIR=ROOT/'evaluations/family_rotation_nested_complementarity_v1/s1_s2_s3_seed42'

!python -u /content/project/scripts/analyze_family_rotation_nested_complementarity.py \
  --transfer-root "{TRANSFER_ROOT}" \
  --output-dir "{OUTPUT_DIR}" \
  --selections S1 S2 S3 \
  --uniform-floor 0.5
```

## Cell 3 — 핵심 결과

```python
import pandas as pd
display(pd.read_csv(OUTPUT_DIR/'selection_summary.csv').round(6))
display(pd.read_csv(OUTPUT_DIR/'overall_summary.csv').round(6))
print((OUTPUT_DIR/'REPORT.md').read_text())
```

판정 기준은 `mean_auc_delta > 0`, `worst_auc_delta`가 크게 음수가 아님,
그리고 6개 holdout 중 다수에서 `wins`가 확인되는 것이다. S1-S3에서 반복되지 않으면
상보성 가중치를 실제 학습 loss로 확장하지 않는다.
