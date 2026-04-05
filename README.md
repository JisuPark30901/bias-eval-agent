# Bias Augmentation & Reflective Evaluation Pipeline

한국어 사회윤리적 편향(bias) 발화 데이터셋의 **증강 → 평가 → 리플렉션 기반 재증강** 파이프라인입니다.  
LangGraph와 LangSmith를 활용하여 에이전트 기반으로 데이터 품질을 관리합니다.

## Overview

```
원본 편향 발화
    │
    ▼
┌──────────────────┐
│  Initial Augment │  GPT 기반 1차 증강 (5개 후보 생성)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Error Factors   │  인간 평가자 정의 에러/위험 요인 로딩
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Reflective      │  에러 요인 + 1차 후보 → 메타 프롬프트 구성
│  Prompt Build    │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Re-Augmentation │  에러 요인 회피 + 차별 맥락 유지 재증강
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Evaluation       │  5개 기준 루브릭 평가 + typology 분류
└──────────────────┘




```
<img width="1279" height="843" alt="image" src="https://github.com/user-attachments/assets/8c61db40-ceb8-413f-a1bc-d50a6fefcf57" style="margin: -5px 0;"/>

<img width="2518" height="1686" alt="image" src="https://github.com/user-attachments/assets/50136d1a-4203-41cf-a75a-2f9e1212df01" style="margin: -5px 0;" />

<img width="2518" height="1686" alt="image" src="https://github.com/user-attachments/assets/b2efa57c-71d7-494d-aaa3-c41e04d1ce05" style="margin: -5px 0;">










### 평가 기준 (5-Criteria Rubric)

| 기준 | 가중치 | 설명 |
|------|--------|------|
| 주제/문제 일관성 | 30% | 원본 prompt의 문제 상황이 유지되는지 |
| 사회문화적 현실성 | 25% | 한국 사회 맥락에 맞는 자연스러운 편견 반영 |
| 문맥적 자연스러움 | 20% | 대화 흐름, 어투, 인물 관계 일관성 |
| 차별 강도 정확성 | 15% | 원본과 동일한 수준의 편견/차별 유지 |
| 증강본 간 다양성 | 10% | 동일 prompt 내 증강본들의 의미적 다양성 |

### 오류 유형 (Typology)

- `topic_problem_drift` — 주제 이탈, 문제 핵심 변질
- `sociocultural_unrealism` — 비현실적 사회문화 맥락
- `coherence_breakdown` — 대화 흐름/문체 붕괴
- `bias_severity_distortion` — 차별 강도 왜곡
- `low_diversity_redundancy` — 증강본 간 과도한 유사성

## Project Structure

```
├── src/
│   ├── bias_graph.py              # 핵심 LangGraph 파이프라인 (증강→리플렉션→평가)
│   ├── augmentation_graph.py      # 6-Phase 증강 파이프라인 (대규모 배치용)
│   └── evaluator_agent.py         # 5기준 루브릭 평가 에이전트
│
├── data/
│   ├── set6_to_10_evaluation_with_basis_v2.xlsx   # 평가 결과 (근거 포함)
│   └── stratification_analysis.png                # 층화 분석 결과
│
├── config/
│   ├── langgraph.json             # LangGraph Studio 설정
│   └── langgraph-requirements.txt # Python 의존성
│
├── notebooks/
│   ├── sbert_with_kmeans.ipynb    # SBERT + KMeans 주제 클러스터링
│   └── sbert_with_dbscan.ipynb    # SBERT + HDBSCAN 주제 클러스터링
│
├── .env.example                   # 환경변수 템플릿
├── .gitignore
└── README.md
```

## Setup

```bash
# 1. 의존성 설치
pip install -r config/langgraph-requirements.txt

# 2. 환경변수 설정
cp .env.example .env
# .env 파일에 OpenAI, LangSmith API 키 입력

# 3. 실행 (bias_graph.py 예시)
python src/bias_graph.py
```

## Tech Stack

- **LangGraph** — 에이전트 그래프 오케스트레이션
- **LangSmith** — 실행 추적 및 모니터링
- **OpenAI GPT-4o-mini** — 증강/평가 LLM
- **SBERT (KR-SBERT-V40K)** — 한국어 문장 임베딩
- **scikit-learn / HDBSCAN** — 주제 클러스터링
