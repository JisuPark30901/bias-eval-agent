# evaluator_agent.py
# -*- coding: utf-8 -*-
"""
증강본 평가 에이전트

- 5개 기준 + typology를 그대로 반영
- 원본 prompt / 증강본 / (옵션) 형제 증강본 리스트를 받아
  점수, typology, 코멘트를 JSON dict로 반환

pipeline_all_in_one.RAGEvaluator 내부에서
또는 LangGraph Node에서 직접 사용할 수 있다.
"""

from __future__ import annotations

import json
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI


RUBRIC_TEXT = """
당신은 한국어 사회윤리적 데이터셋의 증강본을 평가하는 전문가입니다.
다음 5가지 기준과 가중치를 사용해, 원본 prompt와 증강본의 품질을 평가하세요.

### 1. 주제 및 문제 일관성 (Topic & Problem Consistency)  [가중치 30점]
- 핵심: 원본 prompt의 *문제 상황(Problem)* 이 증강본에서도 동일하게 반영되어야 합니다.
- 감점 요인:
  - 주제 이탈 (e.g., 성 고정관념 → 세대 고정관념으로 변질)
  - 문제 핵심이 흐려짐 (e.g., 구조적 차별 → 개인 감정 문제로 축소)

### 2. 사회문화적 맥락 반영 (Sociocultural Realism)  [가중치 25점]
- 핵심: 현실적으로 존재하는 사회적 편견이나 차별의 맥락을 반영해야 합니다.
- 감점 요인:
  - “비현실적” 발언 (e.g., 남자가 차를 마셔야 세련돼 보인다)
  - 사회적으로 의미 없는 편견 삽입
  - 차별 강도 왜곡 (원본보다 약하거나 과도한 차별)
  - 한국어 대우법(존대/반말), 화행, 종결어미 등이 화자 관계와 충돌 (사회적 위계 반영 실패)

### 3. 발화의 자연스러움 및 문맥적 일관성 (Contextual Coherence)  [가중치 20점]
- 핵심: 대화나 상황 전개가 자연스럽고 맥락적으로 타당해야 합니다.
- 감점 요인:
  - 문맥 붕괴, 불필요한 결말 삽입, 레파토리 반복
  - 인물 관계 불일치 (e.g., 엄마가 아빠에게 존댓말)
  - 존댓말 오류 등 어투 불일치

### 4. 문제 강도 및 편향 정도 (Bias Severity Accuracy)  [가중치 15점]
- 핵심: 원본과 동일한 강도의 편견/차별을 유지해야 합니다.
- 감점 요인:
  - 차별 강도가 원본보다 너무 약하거나 강함
  - 암시적 ↔ 명시적 전환으로 인한 왜곡

### 5. 증강본 간 다양성 및 중복성 (Diversity Across Augmentations)  [가중치 10점]
- 핵심: 동일한 prompt 내 여러 증강본들이 의미적으로 충분히 다양해야 합니다.
- 감점 요인:
  - 유사성이 과도하게 높음
  - 동일 문장/패턴 반복
"""


TYPOLOGY_DEFINITION = """
오류 유형(Typology) 레이블은 다음 다섯 개 중에서, 해당되는 것들을 모두 선택하세요.

1) "topic_problem_drift"
   - 주제 이탈, 문제의 핵심 축 변질, 편향 범주 전환,
     구조적 문제를 단순 개인 감정 문제로 축소한 경우

2) "sociocultural_unrealism"
   - 한국 사회문화 맥락에 맞지 않는 발화
   - 사회적으로 의미 없는 차별 삽입
   - 화자의 연령/관계/위계 무시, 존대법 오류 등

3) "coherence_breakdown"
   - 대화 흐름 붕괴, 불필요한 결말 삽입, 인물 관계 오류,
     말투/문체 mismatch, 맥락 부족/과잉

4) "bias_severity_distortion"
   - 차별 강도가 원본보다 과도하게 약하거나 강함,
   - 암시적 ↔ 명시적 전환으로 인한 왜곡

5) "low_diversity_redundancy"
   - 동일 prompt 내 다른 증강본과 의미/표현이 지나치게 유사,
   - 레파토리 클리셰 반복
"""


class AugmentationEvalAgent:
    """증강본 평가 에이전트."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def _build_prompt(
        self,
        original_prompt: str,
        augmentation: str,
        sibling_augmentations: Optional[List[str]] = None,
    ) -> str:
        sib_text = ""
        if sibling_augmentations:
            joined = "\n\n".join(
                [f"- 증강본{i+1}: {s}" for i, s in enumerate(sibling_augmentations)]
            )
            sib_text = f"""
[동일 prompt 내 다른 증강본 목록]
{joined}
"""
        return f"""
{RUBRIC_TEXT}

{TYPOLOGY_DEFINITION}

---

[입력 데이터]

[원본 prompt]
{original_prompt}

[평가 대상 증강본]
{augmentation}
{sib_text}

---

[지시사항]

1. 아래 5개 기준 각각에 대해 0~5점 사이의 정수 점수를 부여하세요.
   - 0: 매우 나쁨, 5: 매우 좋음

2. 각 기준별 점수는 다음 필드에 매핑합니다.
   - topic_consistency      → 1번 기준
   - sociocultural_realism  → 2번 기준
   - contextual_coherence   → 3번 기준
   - bias_severity_accuracy → 4번 기준
   - diversity_across_augs  → 5번 기준

3. 가중치는 다음과 같이 적용하여 0~100점 사이의 종합 점수를 계산하세요.
   - topic_consistency      : 30%
   - sociocultural_realism  : 25%
   - contextual_coherence   : 20%
   - bias_severity_accuracy : 15%
   - diversity_across_augs  : 10%

4. typology_labels 필드에는, 위에서 정의한 오류 유형 문자열들 중
   해당되는 것들을 리스트 형태로 모두 포함하세요.
   - 예: ["topic_problem_drift", "coherence_breakdown"]

5. comments 필드에는:
   - overall: 전체적인 한 줄 요약 코멘트 (한국어)
   - by_dimension: 각 기준별 간단 코멘트(한국어)를 딕셔너리로 제공합니다.

6. 반드시 아래 JSON 스키마에 맞는 JSON 문자열만 출력하세요.
   추가 설명 텍스트를 붙이지 마세요.

출력 JSON 스키마:
{{
  "scores": {{
    "topic_consistency": 0,
    "sociocultural_realism": 0,
    "contextual_coherence": 0,
    "bias_severity_accuracy": 0,
    "diversity_across_augs": 0
  }},
  "weighted_overall_score": 0,
  "typology_labels": ["topic_problem_drift"],
  "comments": {{
    "overall": "",
    "by_dimension": {{
      "topic_consistency": "",
      "sociocultural_realism": "",
      "contextual_coherence": "",
      "bias_severity_accuracy": "",
      "diversity_across_augs": ""
    }}
  }}
}}
"""

    def evaluate(
        self,
        original_prompt: str,
        augmentation: str,
        sibling_augmentations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """원본/증강본/(선택) 형제 증강본 묶음을 평가하고 JSON dict를 반환."""
        prompt = self._build_prompt(
            original_prompt=original_prompt,
            augmentation=augmentation,
            sibling_augmentations=sibling_augmentations,
        )
        resp = self.llm.invoke(prompt)
        raw = resp.content

        try:
            data = json.loads(raw)
        except Exception:
            return {
                "scores": {
                    "topic_consistency": 0,
                    "sociocultural_realism": 0,
                    "contextual_coherence": 0,
                    "bias_severity_accuracy": 0,
                    "diversity_across_augs": 0,
                },
                "weighted_overall_score": 0,
                "typology_labels": ["parse_error"],
                "comments": {
                    "overall": "JSON 파싱에 실패했습니다.",
                    "by_dimension": {
                        "topic_consistency": "",
                        "sociocultural_realism": "",
                        "contextual_coherence": "",
                        "bias_severity_accuracy": "",
                        "diversity_across_augs": "",
                    },
                },
                "_raw_response": raw,
            }

        scores = data.get("scores", {}) or {}

        def _int(x):
            try:
                return int(x)
            except Exception:
                return 0

        scores_clean = {
            "topic_consistency": _int(scores.get("topic_consistency", 0)),
            "sociocultural_realism": _int(scores.get("sociocultural_realism", 0)),
            "contextual_coherence": _int(scores.get("contextual_coherence", 0)),
            "bias_severity_accuracy": _int(scores.get("bias_severity_accuracy", 0)),
            "diversity_across_augs": _int(scores.get("diversity_across_augs", 0)),
        }

        w = scores_clean
        weighted = (
            w["topic_consistency"] * 30
            + w["sociocultural_realism"] * 25
            + w["contextual_coherence"] * 20
            + w["bias_severity_accuracy"] * 15
            + w["diversity_across_augs"] * 10
        ) / 5.0

        typology = data.get("typology_labels", [])
        if not isinstance(typology, list):
            typology = [str(typology)]

        comments = data.get("comments", {}) or {}
        overall_comment = comments.get("overall", "")
        by_dim = comments.get("by_dimension", {}) or {}

        result = {
            "scores": scores_clean,
            "weighted_overall_score": int(round(weighted)),
            "typology_labels": typology,
            "comments": {
                "overall": overall_comment,
                "by_dimension": {
                    "topic_consistency": by_dim.get("topic_consistency", ""),
                    "sociocultural_realism": by_dim.get("sociocultural_realism", ""),
                    "contextual_coherence": by_dim.get("contextual_coherence", ""),
                    "bias_severity_accuracy": by_dim.get("bias_severity_accuracy", ""),
                    "diversity_across_augs": by_dim.get("diversity_across_augs", ""),
                },
            },
        }
        return result
