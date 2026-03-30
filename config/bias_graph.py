from dotenv import load_dotenv
load_dotenv(dotenv_path="data/05_증강관련/.env")

# LangSmith
from langsmith import traceable

from typing import List, Dict, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# ----- 상태 정의 -----
class GraphState(TypedDict, total=False):
    original_prompt: str                   # 원본 발화
    aug_candidates: List[Dict]             # 1차 증강 후보
    error_factors: List[str]               # 인간 평가자가 정의한 에러/위험 요인
    reflective_prompt: str                 # 리플렉트용 메타 프롬프트
    reaug_candidates: List[Dict]           # 재증강 후보
    selected: Optional[Dict]               # 최종 선택 문장
    feedback: Optional[str]                # 평가 코멘트(raw)


llm_augment = ChatOpenAI(model="gpt-4o-mini")
llm_reflect = ChatOpenAI(model="gpt-4o-mini")


# ----- node1: 초기 증강 -----
@traceable
def initial_augment(state: GraphState) -> GraphState:
    prompt = state["original_prompt"]

    sys = SystemMessage(
        content=(
            "너는 한국어 사회윤리적 편향 발화 증강기이다. "
            "원본 발화의 문제 상황(Problem)과 암묵적 차별 맥락을 유지하면서 "
            "현실적인 대화체 문장 5개를 생성하라. "
            "번호를 붙이지 말고 한 줄에 한 문장씩 출력하라."
        )
    )
    human = HumanMessage(content=f"원본 발화:\n{prompt}")

    res = llm_augment.invoke([sys, human])
    lines = [l.strip() for l in res.content.split("\n") if l.strip()]

    aug_candidates = [{"text": t, "score": 0.0} for t in lines]

    return {"aug_candidates": aug_candidates}


# ----- node2: human error factors 로딩 (rule-based placeholder) -----
@traceable
def load_error_factors(state: GraphState) -> GraphState:
    """
    실제로는 사람이 정의한 rule 기반 리스트를 불러오거나,
    사전에 만든 JSON/CSV를 읽어올 수 있는 자리.
    여기서는 원본 발화 내용에 따라 간단 rule만 예시로 둔다.
    """
    prompt = state["original_prompt"]
    factors: List[str] = []

    if "여직원" in prompt or "여자" in prompt:
        factors.append("성별 고정관념(여성=감정적/비합리적)")
    if "남자" in prompt:
        factors.append("성별 역할 고정관념(남성=리더십/중요 업무)")
    if "프레젠테이션" in prompt or "발표" in prompt:
        factors.append("능력 평가를 감정/성별에 귀속시키는 표현")

    # 최소 한 개는 있도록 fallback
    if not factors:
        factors.append("사회적 소수자에 대한 암묵적 폄하 가능성")

    return {"error_factors": factors}


# ----- node3: reflective prompt construction -----
@traceable
def build_reflective_prompt(state: GraphState) -> GraphState:
    """
    1차 증강 후보 + 인간이 정의한 에러 요인을 묶어서
    re-augmentation 및 evaluator가 참고할 메타 프롬프트를 만든다.
    """
    prompt = state["original_prompt"]
    cands = state.get("aug_candidates", [])
    factors = state.get("error_factors", [])

    joined_cands = "\n".join(f"- {c['text']}" for c in cands)
    joined_factors = "\n".join(f"- {f}" for f in factors)

    reflective_prompt = (
        "다음은 원본 발화와 1차 증강 후보, 그리고 인간 평가자가 사전에 정의한 "
        "에러/위험 요인 목록이다.\n\n"
        f"[원본 발화]\n{prompt}\n\n"
        f"[1차 증강 후보]\n{joined_cands}\n\n"
        "[인간 에러 요인(Human error factors)]\n"
        f"{joined_factors}\n\n"
        "위 정보를 참고하여, 에러 요인을 회피하거나 완화하면서도 "
        "여전히 원본 발화의 문제 상황과 암묵적 차별 맥락을 유지하도록 "
        "재증강을 수행하라."
    )

    return {"reflective_prompt": reflective_prompt}


# ----- node4: re-augmentation -----
@traceable
def re_augment(state: GraphState) -> GraphState:
    """
    node3에서 만든 reflective_prompt를 기반으로 재증강 수행.
    """
    rprompt = state["reflective_prompt"]

    sys = SystemMessage(
        content=(
            "너는 인간 평가자가 정의한 에러 요인을 참고하여 "
            "편향 발화를 보다 정교하게 재증강하는 한국어 증강기이다. "
            "에러 요인을 피하면서도 문제 상황/차별 맥락은 유지하라. "
            "현실적인 대화체 문장 5개를 생성하라. "
            "번호를 붙이지 말고 한 줄에 한 문장씩 출력하라."
        )
    )
    human = HumanMessage(content=rprompt)

    res = llm_augment.invoke([sys, human])
    lines = [l.strip() for l in res.content.split("\n") if l.strip()]

    reaug_candidates = [{"text": t, "score": 0.0} for t in lines]

    return {"reaug_candidates": reaug_candidates}


# ----- node5: reflective evaluator -----
@traceable
def reflective_evaluate(state: GraphState) -> GraphState:
    """
    재증강 후보들에 대해 human error factors를 기준으로 평가하는 evaluator.
    최종 선택까지 여기서 수행.
    """
    prompt = state["original_prompt"]
    cands = state["reaug_candidates"]
    factors = state.get("error_factors", [])

    joined_cands = "\n".join(f"{i+1}. {c['text']}" for i, c in enumerate(cands))
    joined_factors = "\n".join(f"- {f}" for f in factors)

    sys = SystemMessage(
        content=(
            "너는 한국어 마이크로어그레션 평가자이다. "
            "인간 평가자가 정의한 에러 요인을 기준으로, "
            "각 후보가 (1) 주제/문제 일관성, (2) 사회문화적 현실성, "
            "(3) 암묵적 차별 강도, (4) 표현 다양성, (5) 에러 요인 회피 정도 측면에서 "
            "얼마나 좋은지 0~5점으로 평가하고, 총합 점수를 계산하라. "
            "JSON 리스트 형태로만 답하라. "
            "형식: [{\"idx\": 1, \"score\": 17.0, \"comment\": \"...\"}, ...]"
        )
    )
    human = HumanMessage(
        content=(
            f"원본 발화:\n{prompt}\n\n"
            f"재증강 후보들:\n{joined_cands}\n\n"
            "[인간 에러 요인 목록]\n"
            f"{joined_factors}"
        )
    )

    res = llm_reflect.invoke([sys, human])
    import json

    evals = json.loads(res.content)

    scored = []
    for e in evals:
        i = e["idx"] - 1
        base = cands[i].copy()
        base["score"] = float(e["score"])
        base["comment"] = e.get("comment", "")
        scored.append(base)

    # 최종 선택
    best = max(scored, key=lambda c: c.get("score", 0.0)) if scored else None

    return {
        "reaug_candidates": scored,
        "selected": best,
        "feedback": res.content,
    }


# ----- 그래프 컴파일 -----
graph = StateGraph(GraphState)

graph.add_node("initial_augment", initial_augment)
graph.add_node("load_error_factors", load_error_factors)
graph.add_node("build_reflective_prompt", build_reflective_prompt)
graph.add_node("re_augment", re_augment)
graph.add_node("reflective_evaluate", reflective_evaluate)

graph.set_entry_point("initial_augment")
graph.add_edge("initial_augment", "load_error_factors")
graph.add_edge("load_error_factors", "build_reflective_prompt")
graph.add_edge("build_reflective_prompt", "re_augment")
graph.add_edge("re_augment", "reflective_evaluate")
graph.add_edge("reflective_evaluate", END)

workflow = graph.compile()


# ----- 로컬 테스트 -----
if __name__ == "__main__":
    from pprint import pprint

    state: GraphState = {
        "original_prompt": "여직원은 감정 기복이 심해서 중요한 프레젠테이션은 남자 직원이 하는 게 낫지 않겠어?"
    }
    result = workflow.invoke(state)
    pprint(result)
