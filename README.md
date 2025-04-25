# OS Project: CNN 연산 병렬 처리 성능 비교 실험

본 프로젝트는 **운영체제 수업**의 일환으로, CNN(Convolutional Neural Network) 연산을 다양한 병렬화 구조(Process 기반, Thread 기반, Hybrid 구조)로 구현하고  
멀티코어 환경에서의 성능을 **정량적으로 측정 및 분석**하는 것을 목표로 합니다.

---

## 📌 전체 연산 구조
Input → Conv2D → ReLU → MaxPool2D → Flatten → FullyConnected → Softmax → Output

- 입력값, weight, bias는 단순 초기화

---

## 📅 수행 단계별 계획

### 1. 레이어 구조 정의
- 전체 CNN 모델을 계층별로 구성

### 2. 레이어별 병렬화 구조 실험
| 조건 | 설명 |
|------|------|
| (가) | 단일 thread만 사용 |
| (나) | 단일 child process + thread 사용 |
| (다) | 다중 process + 다중 thread |
| (라) | Hybrid 구조 (적절히 multiprocess와 multithread 사용) |
| (마) | sync 유무 비교 |

### 3. GitHub 공동 작업
- 브랜치 분리: `main`, 개인 브랜치
- PR + 리뷰 기반 코드 병합

---

## 🧪 실험 계획

- Step 1: Baseline CNN 단일 구조 구현
- Step 2: Conv2D 계층 병렬화 (5가지 조건)
- Step 3: FullyConnected 계층 병렬화
- Step 4: 최적 구조 조합 CNN 구현
- Step 5: Baseline과의 성능 비교

> ⚠️ 실험 중 성능 개선 아이디어가 나오면 최적화 추가 가능

---

## 📏 측정 항목 (Measurement)

### 기본 측정
- 실행 시간 (real, user, sys)
- 메모리 사용량 (RSS)
- CPU core 활용률 (`htop`, `mpstat`)

### 병렬 구조 분석 지표
- `perf stat -e context-switches`
- `perf stat -e cache-misses`
- `perf stat -e cpu-migrations`
- IPC 계산 = instructions / cycles
- Scaling efficiency
- Shared memory 사용 여부

---

## 📁 폴더 구조 

## 📁 폴더 구조

```
OSproject/
├── README.md           # 프로젝트 설명
├── .gitignore          # 실행파일, 중간 빌드 결과 제외

├── /src                # CNN 구조 및 병렬화 구현 소스
│   ├── baseline/       # 단일 스레드 구현
│   ├── pthread/        # pthread 기반 병렬 처리
│   ├── fork/           # fork 기반 병렬 처리
│   └── hybrid/         # fork + pthread 혼합 구조

├── /include            # 공통 헤더 파일
│   └── cnn.h           # 함수 선언, 구조체 등

└── /bin                # 빌드된 실행 파일 저장
    ├── baseline.out
    ├── pthread.out
    ├── fork.out
    └── hybrid.out
```

---

