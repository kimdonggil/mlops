# 🧠 MLOps 설치 & 실습 가이드

이 프로젝트는 NVIDIA GeForce RTX 3090 GPU 0번을 활용한 **MLOps 설치 & 실습 가이드**입니다. Docker 설치, Kubernetes 설정, Kubeflow 설치 등을 포함합니다.

&nbsp;

# 🖥️ 서버 정보

- **GPU**: NVIDIA GeForce RTX 3090(24GB)
- **OS**: Ubuntu 24.04.1 LTS
- **CUDA**: 12.2
- **NVIDIA Driver**: 535.230.02
- **Python**: 3.12.3
- **환경 관리**: Container, Kubernetes

&nbsp;

# 📂 프로젝트 구조

```bash

guide/
├── part 1
│   ├── 1.1 Docker Engine 설치
│   ├── 1.2 Docker Compose 설치
│   ├── 1.3 NVIDIA Docker 설치
├── part 2
│   ├── 2.1 Kubernetes 초기 설정
│   ├── 2.2 Kubernetes Cluster 설정
│   ├── 2.3 Kubernetes Cluster 삭제
├── part 3
│   ├── 3.1 Kubeflow 설치
│   ├── 3.2 Kubeflow 실행
│   ├── 3.3 Kubeflow 삭제
│   ├── 3.4 Kubeflow Volumes 설정
│   ├── 3.5 Kubeflow 이슈
├── part 4
│   ├── 예제 데이터
│   │   └── 4.1 MinIO Browser 예제 데이터
│   └── 4.1 MinIO Browser 예제
├── part 5
│   ├── 5.1 KFP 설치
│   ├── 5.2 KFP YOLO 예제
│   ├── 5.3 KFP 이슈
└── part 6
    ├── 6.1 BentoML 설치 
    ├── 6.2 BentoML YOLO 예제
    └── 6.3 BentoML 실행

```

`part 1`은 Docker 및 NVIDIA Container 환경 구축 가이드로, 기초적인 Docker 엔진 설치부터 NVIDIA GPU를 활용하기 위한 Docker 설정까지 단계별로 설명합니다.

`part 2`는 Kubernetes Cluster 환경 구성 및 관리 가이드로, Cluster 초기 세팅, 운영 및 삭제 방법까지 포함된 Kubernetes 환경 구축 방법을 설명합니다.

`part 3`은 Kubeflow Pipeline 구축 및 운영 가이드로, Kubeflow 설치부터 실행, 삭제 및 스토리지 설정 등의 방법을 설명하고, 자주 발생하는 이슈도 함께 정리되어 있습니다.

`part 4`는 MinIO 예제로, MinIO Browser를 활용한 데이터 업로드, 다운로드 등을 실습합니다.

`part 5`는 KFP(Kubeflow Pipelines) 예제로, KFP 설치 방법부터 YOLO 모델을 활용한 Pipeline 예제를 통해 KFP 사용 방법을 실습합니다.

`part 6`은 BentoML 예제로, BentoML 설치부터 YOLO 모델 훈련 예제를 통해 BentoML 사용 방법을 실습합니다.
