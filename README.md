# OS Project: CNN Inferece 연산 병렬 처리 성능 비교 실험

본 프로젝트는 **운영체제 수업**의 일환으로, CNN(Convolutional Neural Network) Inference 연산을 다양한 병렬화 구조(Process 기반, Thread 기반, Hybrid 구조)로 구현하고  
멀티코어 환경에서의 성능을 **정량적으로 측정 및 분석**하는 것을 목표로 합니다.

---

## 📌 전체 연산 구조
Input → Conv2D → ReLU → MaxPool2D → Flatten → FullyConnected → Softmax → Output

- 입력값은 9개의 stream으로 처리하며, 순서대로 1~9로 이루어진 행렬 값을 32x32 image로 간주
- weight, bias는 짝수 행은 1, 홀수 행은 0으로 단순 반복 값 초기화

---

## 🧪 실험 계획

- Step 1: Baseline CNN 단일 구조 구현
- Step 2: Single Child & Thread 구조 구현
- Step 3: Multi Child & Thread 구조 구현
- Step 4: 최적화 구조 구현
    - Single Child + Multi Thread
    - Multi Child + Multi Thread
- Step 5: Baseline CNN과의 성능 비교
- Step 6: Synchronization 유무에 따른 결과 비교

---

## 📏 측정 항목 (Measurement)

### 기본 측정
- 실행 시간 (real, user, sys)
- 메모리 사용량 (RSS)
- Context Switching
- Page Fault

### 병렬 구조 분석 지표
- `perf stat -e cache-misses`
- `perf stat -e cpu-migrations`

---

## 📁 폴더 구조 


```
OSproject/
├── README.md           # 프로젝트 설명 및 계획
├── .gitignore          # 실행파일, 중간 빌드 결과 제외 설정

├── /src                # CNN 기반 모델 전체 소스 코드
│   ├── BaselineCNN.c   # 기본 구현 코드
│   ├── singleChild/    # fork 기반 단일 child process 처리 구조
│   ├── singleThread/   # pthread 기반 단일 thread 처리 구조
│   ├── multiChild/     # multi child process 병렬 처리 구조
│   ├── multiThread/    # multi thread 병렬 처리 구조
│   ├── opt/            # 성능 최적화 코드 (fork & pthread 병렬 처리 구조)
│   └── sync/           # 동기화 유무에 따른 비교

├── /include            # 공통 헤더 파일
    └── layers.h        # model layer 구조체, 연산 함수 선언 및 정의ㅣ

---

