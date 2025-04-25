# OS Project: CNN 연산 병렬 처리 성능 비교 실험

본 프로젝트는 **운영체제 수업**의 일환으로, CNN(Convolutional Neural Network) 연산을 다양한 병렬화 구조(Process 기반, Thread 기반, Hybrid 구조)로 구현하고  
멀티코어 환경에서의 성능을 **정량적으로 측정 및 분석**하는 것을 목표로 합니다.

---

## 📌 전체 연산 구조
Input → Conv2D → ReLU → MaxPool2D → Flatten → FullyConnected → Softmax → Output

- 입력값, weight, bias는 단순 초기화
- 입력 데이터: 무작위 생성 또는 CIFAR10 추출 텍스트 사용

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
| (라) | Hybrid 구조 |
| (마) | sync 유무 비교 (mutex, barrier 등)

### 3. 역할 분담
- 각자 CNN의 특정 계층/병렬 구조/측정 항목 담당

### 4. GitHub 공동 작업
- 브랜치 분리: `main`, `dev`, 개인 브랜치 (`seo-conv2d`, `hje-thread`, …)
- PR + 리뷰 기반 코드 병합

---

## 🧪 실험 계획

- Step 1: Baseline CNN 단일 구조 구현
- Step 2: Conv2D 계층 병렬화 (5가지 조건)
- Step 3: MaxPool2D 계층 병렬화
- Step 4: FullyConnected 계층 병렬화
- Step 5: 최적 구조 조합 CNN 구현
- Step 6: Baseline과의 성능 비교

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

## 📁 폴더 구조 (예정)
/src         → CNN 구조 및 병렬화 구현
/include     → 헤더 파일
/data        → 입력값 텍스트 파일
/results     → 측정 결과 로그
/plots       → 그래프 이미지
README.md    → 프로젝트 설명서

---

## 👨‍👩‍👧‍👦 팀 구성

| 이름 | 역할 |
|------|------|
| 서아 | Conv2D, fork 구조 |
| 화정 | FC layer, pthread 병렬화 |
| 수빈 | 동기화 실험, shared memory |
| 서연 | 실험 자동화, 측정 스크립트 작성

---

## 🔒 브랜치 전략

- `main`: 최종 제출용 (직접 push 금지)
- `dev`: 실험 통합 브랜치
- `개인 브랜치`: 기능별 개발 후 PR로 `dev`에 병합
