# 🧠 D-Lab Flow

&nbsp;

# 🖥️ 서버 정보

- **GPU**: NVIDIA H100(80GB)
- **OS**: Ubuntu 24.04.1 LTS
- **CUDA**: 12.2
- **NVIDIA Driver**: 535.183.01
- **Python**: 3.10.18
- **환경 관리**: Container, Kubernetes

&nbsp;

# 📂 프로젝트 구조

```bash

dlabflow/
├── bentoml
│   ├── service.py
└── pipline
    ├── inference.py
    ├── preprocessing.py
    └── training.py

```

`bentoml/service.py` AI 워크플로우를 처리하기 위한 BentoML 서비스 정의 파일입니다.

`pipeline/inference.py` 학습된 모델을 사용한 추론 로직을 정의한 파일입니다.

`pipeline/preprocessing.py` 데이터 전처리 로직을 정의한 파일입니다.

`pipeline/training.py` 모델 학습 및 평가 관련 로직을 정의한 파일입니다.