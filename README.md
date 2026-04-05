# Bias Augmentation & Reflective Evaluation Pipeline

한국어 사회윤리적 편향(bias) 발화 데이터셋의 **증강 → 평가 → Reflective Agent 기반 재증강** 파이프라인입니다.  
LangGraph와 LangSmith를 활용하여 에이전트 기반으로 데이터 품질을 관리합니다.

## Overview
<img width="2000" height="1125" alt="portfolio_selectstar_260403-2" src="https://github.com/user-attachments/assets/c14052be-af7a-4b23-a34f-61fd2ad14ad0" />
<img width="2000" height="1125" alt="portfolio_selectstar_260403-3" src="https://github.com/user-attachments/assets/d7ba3f2a-96d6-4c61-b202-279c610821ed" />
<img width="2000" height="1125" alt="portfolio_selectstar_260403-4" src="https://github.com/user-attachments/assets/29f7d995-d8d6-4aff-9fff-8527cb4e5364" />

<img width="2000" height="1125" alt="portfolio_selectstar_260403-5" src="https://github.com/user-attachments/assets/5796cb50-36a4-45cb-bf21-820abe498445" />
<img width="2000" height="1125" alt="portfolio_selectstar_260403-6" src="https://github.com/user-attachments/assets/a6a55df3-b431-4922-abd4-48f781639a00" />
<img width="2000" height="1125" alt="portfolio_selectstar_260403-7" src="https://github.com/user-attachments/assets/0a58991c-eef2-4c87-8311-0710cd388d0e" />


```
원본 편향 발화
    │
    ▼
┌──────────────────┐
│  Initial Augment │  GPT 기반 1차 증강 (5개 후보 생성)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Error Analysis  │  인간 평가자 정의 에러/위험 요인 로딩
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Prompt Re- Build │  에러 요인 + 1차 후보 → 메타 프롬프트 구성
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Re-Augmentation │  에러 요인 회피 + 차별 맥락 유지 재증강
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Evaluation      │  5개 기준 루브릭 평가 + typology 분류
└──────────────────┘




```


<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h2>Motivation: Why Not LLM-as-a-Judge Alone? & 기존 벤치마크의 한계 </h2>

<p>
  ROUGE, BLEU와 같은 전통적 지표가 n-gram 중복만을 측정해 텍스트의 의미적 품질을 포착하지 못하는 한계를 가지듯,
  LLM-as-a-Judge 역시 독립적인 평가 방법으로 사용될 경우 신뢰성과 공정성 측면에서 구조적 한계를 가진다.
  아래 표는 이러한 한계를 실증적으로 보고한 주요 연구들을 정리한 것으로, 검토 방식으로 Human-in-the-Loop 방식을 채택한 근거가 된다.
</p>

<h3>📌 Bias</h3>
<table>
  <thead>
    <tr>
      <th>논문</th>
      <th>Venue</th>
      <th>연도</th>
      <th>한계 유형</th>
      <th>핵심 내용</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Zheng et al., <em>Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena</em></td>
      <td>NeurIPS</td>
      <td>2023</td>
      <td>Position / Verbosity / Self-enhancement Bias</td>
      <td>GPT-4도 80% 이상 인간과 일치하지만 편향 완화 없이는 공정하지 않음</td>
    </tr>
    <tr>
      <td>Ye et al., <em>Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge</em></td>
      <td>ICLR</td>
      <td>2025</td>
      <td>12가지 편향 유형</td>
      <td>
        CALM 프레임워크로 verbosity, fallacy oversight, sentiment bias 등 12종 체계화.
        내용 품질이 아닌 표현 스타일·순서 등 표면적 요소에 의해 판단이 달라지며,
        Claude-3.5-Sonnet도 분노 표현 추가만으로 판정을 번복하는 사례 확인.
        GPT-4 포함 6개 모델 실험에서 고성능 모델도 특정 태스크에서 편향 지속.
      </td>
    </tr>
    <tr>
      <td>Wataoka et al., <em>Self-Preference Bias in LLM-as-a-Judge</em></td>
      <td>arXiv</td>
      <td>2024</td>
      <td>Self-preference Bias</td>
      <td>LLM은 perplexity 낮은 출력을 자신이 생성하지 않아도 더 높게 평가</td>
    </tr>
    <tr>
      <td>Shi et al., <em>A Systematic Study of Position Bias in LLM-as-a-Judge</em></td>
      <td>IJCNLP</td>
      <td>2025</td>
      <td>Position Bias</td>
      <td>judge 모델 선택이 positional bias에 가장 큰 영향, 기존 완화 전략도 완전 제거 실패</td>
    </tr>
  </tbody>
</table>

<h3>📌 Consistency / Reliability</h3>
<table>
  <thead>
    <tr>
      <th>논문</th>
      <th>Venue</th>
      <th>연도</th>
      <th>한계 유형</th>
      <th>핵심 내용</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Schroeder &amp; Wood-Doughty, <em>Rating Roulette</em></td>
      <td>EMNLP Findings</td>
      <td>2025</td>
      <td>Intra-rater Inconsistency</td>
      <td>반복 평가 시 Krippendorff's α가 기준치 0.8 미달, temperature=0은 오히려 성능 저하</td>
    </tr>
    <tr>
      <td>Li et al., <em>Can You Trust LLM Judgments?</em></td>
      <td>arXiv</td>
      <td>2024</td>
      <td>IRR 불안정성</td>
      <td>random seed 변동만으로 IRR이 0.167~1.00까지 변동, IRR 자체가 신뢰 지표로 부적합</td>
    </tr>
    <tr>
      <td>Li et al., <em>An Empirical Study of LLM-as-a-Judge</em></td>
      <td>arXiv</td>
      <td>2025</td>
      <td>Score Consistency</td>
      <td>루브릭 구성·점수 기술 방식에 따라 Krippendorff's α 기반 consistency가 크게 달라짐</td>
    </tr>
  </tbody>
</table>

<h3>📌 Domain / Generalization Gap</h3>
<table>
  <thead>
    <tr>
      <th>논문</th>
      <th>Venue</th>
      <th>연도</th>
      <th>한계 유형</th>
      <th>핵심 내용</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Szymanski et al., <em>Limitations of LLM-as-a-Judge in Expert Knowledge Tasks</em></td>
      <td>ACM IUI</td>
      <td>2025</td>
      <td>Expert Domain Gap</td>
      <td>전문 도메인에서 위험하거나 부정확한 내용을 간과하고 지시 따르기에만 집중</td>
    </tr>
    <tr>
      <td>Gu et al., <em>LLMs-as-Judges: A Comprehensive Survey</em></td>
      <td>arXiv</td>
      <td>2024</td>
      <td>다차원적 한계 종합</td>
      <td>프롬프트 템플릿 민감성, 학습 데이터 편향 계승, 도메인별 기준 적용 실패</td>
    </tr>
    <tr>
      <td>Liang et al., <em>A Survey on LLM-as-a-Judge</em></td>
      <td>arXiv</td>
      <td>2024</td>
      <td>다국어/일반화 한계</td>
      <td>다국어 환경에서 Fleiss' κ ≈ 0.3, 모델 크기 확장으로도 해소 안 됨</td>
    </tr>
  </tbody>
</table>

<h3>📌 기존 유해성 측정 관련 데이터셋 </h3>

| 벤치마크 이름 | 연도 | 주요 평가 영역 | 방법론 특징 | 데이터 규모 | 주요 한계점 |
| --- | --- | --- | --- | --- | --- |
| **HarmBench** | 2024 | 유해성, 안전성 (레드티밍) | 자동화된 레드티밍 프레임워크, 510개 유해 행동 | 510개 유해 행동 템플릿 | 새로운 탈옥 기법 반영 어려움 |
| **SALAD-Bench** | 2024 | 유해성, 안전성 (계층적) | 6도메인, 16작업, 66카테고리 계층적 분류 | 21,000+ 유해 질문 | 평가 비용 높음, 문화적 특수성 부족 |
| **AILuminate** | 2025 | 유해성, 안전성 (범용) | 12개 위험 카테고리, 표준화된 등급 산정 | 59,624 프롬프트 | 전문 분야 미세 윤리 문제 포착 어려움 |
| **S-Eval** | 2024 | 유해성, 안전성 (동적) | 자동 및 적응형 테스트 생성 | 동적 생성 테스트 세트 | 결과 재현성 확보 과제 |
| **BBQ** | 2022 | 사회적 편향 (QA) | QA 형식, 모호/명확 문맥 비교 | 58,492 예제 | 객관식 형식의 한계 |
| **BOLD** | 2021 | 사회적 편향 (생성) | 개방형 텍스트 생성 후 감성/독성 분석 | 23,679 프롬프트 | 평가 도구 자체의 편향 가능성 |
| **HolisticBias** | 2022 | 사회적 편향 (포괄적) | 13개 인구통계 축, 600개 정체성 설명자 | 459,758 프롬프트 | 평가 자원 소모, 템플릿 기반 한계 |
| **WinoBias / WinoGender** | 2018 | 사회적 편향 (대명사) | 대명사 상호참조 해결 과제 | 수천 개 문장 쌍 | 성별 이진법 기반, 다양한 정체성 반영 한계 |
| **KoBBQ** | 2024 | 한국어 편향 (QA) | BBQ 한국어 현지화, 문화적 이슈 반영 | 76,048 예제 | 객관식 형식의 한계, 변화하는 편향 반영 어려움 |
| **KoSBi** | 2023 | 한국어 편향 (생성/안전성) | 한국 사회 15개 카테고리, 72개 그룹 | 34,000 문장 쌍 | 인간 평가자 주관 개입, 편향 정의 고정 위험 |
| **KorNAT** | 2024 | 한국어 정렬 (가치/상식) | 한국 사회적 가치 및 상식 정렬 평가 | 10,000 질문 | '한국적 가치' 개념의 다원성 |
| **XSafety / IndicSafe** | 2024 | 다국어 안전성 | 다국어 환경 안전성 평가, 문화적 금기 반영 | 수십만 개 프롬프트 | 저자원 언어 데이터 품질, 문화적 차이 정량화 어려움 |


</body>
</html>






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


## 프로젝트 상세

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
