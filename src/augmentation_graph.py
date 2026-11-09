# graphs/augmentation_graph.py
# -*- coding: utf-8 -*-
"""
LangGraph 기반 증강 파이프라인 그래프

Phase 0: 증강본 전문가 평가
Phase 1: typology 추출
Phase 2: typology 기반 평가 요약
Phase 3: 재증강 프롬프트 구성
Phase 4: 재증강 (generation loop)
Phase 5: 최종 병합/평가 및 저장

실행 순서: 0 → 1 → 2 → 3 → 4 → 5
"""

from __future__ import annotations

from typing import TypedDict, Dict, Any, List
from pathlib import Path
import json

import pandas as pd
from langgraph.graph import StateGraph, START, END
from langsmith import traceable

import pipeline_all_in_one as core


class PipelineState(TypedDict, total=False):
    # 설정값
    config: Dict[str, Any]

    # Phase 0
    df: pd.DataFrame
    df_eval: pd.DataFrame
    good_df: pd.DataFrame
    bad_df: pd.DataFrame

    # Phase 1
    improved_prompt: str
    guidelines: List[str]
    issue_counts: Dict[str, int]

    # Phase 2
    critic_summary: str

    # Phase 3
    regen_prompt: str

    # Phase 4
    regen_df: pd.DataFrame

    # Phase 5
    final_df: pd.DataFrame
    stats: Dict[str, Any]
    quality_report: str


def _merged_config(state: PipelineState) -> Dict[str, Any]:
    """
    state.config를 가져와 기본값과 merge.
    """
    base = {
        "parsed_csv": core.DEFAULT_PARSED_CSV,
        "memo": "",
        "target": core.TARGET_DEFAULT,
        "dry_run": False,
    }
    user_cfg = state.get("config") or {}
    base.update(user_cfg)
    state["config"] = base
    return base


def summarize_state(state: PipelineState) -> Dict[str, Any]:
    """
    LangSmith span에서 보기 좋은 가벼운 state 요약.
    """
    summary: Dict[str, Any] = {}
    if "df" in state:
        summary["df_rows"] = int(len(state["df"]))
    if "good_df" in state:
        summary["good_rows"] = int(len(state["good_df"]))
    if "bad_df" in state:
        summary["bad_rows"] = int(len(state["bad_df"]))
    if "regen_df" in state:
        summary["regen_rows"] = int(len(state["regen_df"]))
    if "final_df" in state:
        summary["final_rows"] = int(len(state["final_df"]))
    if "stats" in state:
        summary["stats"] = state["stats"]
    return summary


@traceable(name="Phase0_ExpertEval")
def phase0_expert_eval(state: PipelineState) -> PipelineState:
    """
    Phase 0.
    - CSV 로드 및 컬럼 정규화
    - (선택) vectorstore 생성
    - RAGEvaluator로 전체 평가
    - 컷라인 기준 good/bad 분리
    """
    cfg = _merged_config(state)
    csv_path = cfg["parsed_csv"]
    memo_path = cfg.get("memo", "")
    dry_run = bool(cfg.get("dry_run", False))

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"[Phase 0] CSV 파일 없음: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = core._normalize_columns(df)
    state["df"] = df

    vs = core._maybe_build_vectorstore(df, memo_path) if core.USE_VECTOR_DB else None
    evaluator = core.RAGEvaluator(use_llm=(not dry_run), vectorstore=vs)

    df_eval = evaluator.evaluate_all(df)
    good_df, bad_df = core.filter_good_bad(df_eval)

    state["df_eval"] = df_eval
    state["good_df"] = good_df
    state["bad_df"] = bad_df

    print("[Phase0 summary]", summarize_state(state))
    return state


@traceable(name="Phase1_TypologyExtraction")
def phase1_typology_extraction(state: PipelineState) -> PipelineState:
    """
    Phase 1.
    - bad_df의 이슈 태그 분포를 기반으로 typology 추출
    - good_df 샘플을 활용해 improved_prompt, guidelines, issue_counts 도출
    """
    good_df = state.get("good_df")
    bad_df = state.get("bad_df")

    if good_df is None or bad_df is None:
        raise RuntimeError("[Phase 1] good_df / bad_df가 없음. Phase 0 선행 필요.")

    improved_prompt, guidelines, issue_counts = core.analyze_and_prompt(good_df, bad_df)

    state["improved_prompt"] = improved_prompt
    state["guidelines"] = guidelines
    state["issue_counts"] = dict(issue_counts)

    print("[Phase1 summary]", summarize_state(state))
    return state


@traceable(name="Phase2_TypologyCriticSummary")
def phase2_typology_based_eval(state: PipelineState) -> PipelineState:
    """
    Phase 2.
    - issue_counts와 guidelines를 바탕으로
      평가/재증강에 참고할 메타 요약 텍스트 생성.
    """
    issue_counts = state.get("issue_counts", {})
    guidelines = state.get("guidelines", [])

    lines = ["[Typology 기반 평가 요약]"]
    if issue_counts:
        lines.append("주요 오류 유형 분포:")
        for k, v in issue_counts.items():
            lines.append(f"  - {k}: {v}회")
    if guidelines:
        lines.append("")
        lines.append("가이드라인:")
        for g in guidelines:
            lines.append(f"  - {g}")

    critic_summary = "\n".join(lines)
    state["critic_summary"] = critic_summary

    print("[Phase2 summary]", summarize_state(state))
    return state


@traceable(name="Phase3_RegenPrompt")
def phase3_build_regen_prompt(state: PipelineState) -> PipelineState:
    """
    Phase 3.
    - Phase 1의 improved_prompt에
      Phase 2의 critic_summary를 덧붙여 최종 regen_prompt 생성.
    """
    base_prompt = state.get("improved_prompt", "")
    critic_summary = state.get("critic_summary", "")

    if not base_prompt:
        raise RuntimeError("[Phase 3] improved_prompt가 비어 있음. Phase 1 선행 필요.")

    if critic_summary:
        regen_prompt = critic_summary + "\n\n" + base_prompt
    else:
        regen_prompt = base_prompt

    state["regen_prompt"] = regen_prompt
    # regenerate_to_target 시그니처를 맞추기 위해 improved_prompt도 갱신
    state["improved_prompt"] = regen_prompt

    print("[Phase3 summary]", summarize_state(state))
    return state


@traceable(name="Phase4_Regeneration")
def phase4_regeneration(state: PipelineState) -> PipelineState:
    """
    Phase 4.
    - regen_prompt(improved_prompt)를 사용해 부족분 재증강.
    - 컷라인 통과 문장만 regen_df에 쌓음.
    """
    cfg = _merged_config(state)
    good_df = state.get("good_df")
    bad_df = state.get("bad_df")
    improved_prompt = state.get("improved_prompt")

    if good_df is None or bad_df is None or improved_prompt is None:
        raise RuntimeError("[Phase 4] good_df / bad_df / improved_prompt 중 일부 없음.")

    dry_run = bool(cfg.get("dry_run", False))
    memo_path = cfg.get("memo", "")

    vs = core._maybe_build_vectorstore(state["df"], memo_path) if core.USE_VECTOR_DB else None
    evaluator = core.RAGEvaluator(use_llm=(not dry_run), vectorstore=vs)

    target_total = int(cfg.get("target", core.TARGET_DEFAULT))

    regen_df = core.regenerate_to_target(
        bad_df=bad_df,
        improved_prompt=improved_prompt,
        evaluator=evaluator,
        target_total=target_total,
        current_good=good_df,
    )

    state["regen_df"] = regen_df

    print("[Phase4 summary]", summarize_state(state))
    return state


@traceable(name="Phase5_Finalize")
def phase5_finalize(state: PipelineState) -> PipelineState:
    """
    Phase 5.
    - good_df + regen_df를 합쳐 중복 제거 및 점수 기반 정렬.
    - target 개수만 남겨 final_df 생성.
    - stats, quality_report 생성 및 파일 저장.
    """
    cfg = _merged_config(state)
    good_df = state.get("good_df")
    regen_df = state.get("regen_df")

    if good_df is None:
        raise RuntimeError("[Phase 5] good_df가 없음. Phase 0 선행 필요.")
    if regen_df is None:
        regen_df = pd.DataFrame(columns=good_df.columns)

    target_total = int(cfg.get("target", core.TARGET_DEFAULT))

    final_df = core.finalize_dataset(
        good_df=good_df,
        new_df=regen_df,
        target=target_total,
    )
    state["final_df"] = final_df

    stats = {
        "최종_개수": int(len(final_df)),
        "평균_현실성": float(final_df["현실성"].mean()) if len(final_df) else 0.0,
        "평균_충실도": float(final_df["충실도"].mean()) if len(final_df) else 0.0,
        "평균_차별강도": float(final_df["차별강도"].mean()) if len(final_df) else 0.0,
        "good_개수": int(len(good_df)),
        "재증강_개수": int(len(regen_df)) if len(regen_df) > 0 else 0,
        "가이드라인": state.get("guidelines", []),
        "주요문제": state.get("issue_counts", {}),
        "target": target_total,
    }
    state["stats"] = stats

    report_text = (
        f"최종 개수: {len(final_df)} / 목표 {target_total}\n"
        f"Good(초기): {stats['good_개수']}\n재증강: {stats['재증강_개수']}\n"
        f"평균 현실성: {stats['평균_현실성']:.2f}\n"
        f"평균 충실도: {stats['평균_충실도']:.2f}\n"
        f"평균 차별강도: {stats['평균_차별강도']:.2f}\n"
    )
    state["quality_report"] = report_text

    final_df.to_csv("final_500.csv", index=False, encoding="utf-8-sig")
    Path("pipeline_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path("quality_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    print("[Phase5 summary]", summarize_state(state))
    return state


graph = StateGraph(PipelineState)

graph.add_node("phase0_expert_eval", phase0_expert_eval)
graph.add_node("phase1_typology_extraction", phase1_typology_extraction)
graph.add_node("phase2_typology_based_eval", phase2_typology_based_eval)
graph.add_node("phase3_build_regen_prompt", phase3_build_regen_prompt)
graph.add_node("phase4_regeneration", phase4_regeneration)
graph.add_node("phase5_finalize", phase5_finalize)

# 실행 순서: 0 → 1 → 2 → 3 → 4 → 5
graph.add_edge(START, "phase0_expert_eval")
graph.add_edge("phase0_expert_eval", "phase1_typology_extraction")
graph.add_edge("phase1_typology_extraction", "phase2_typology_based_eval")
graph.add_edge("phase2_typology_based_eval", "phase3_build_regen_prompt")
graph.add_edge("phase3_build_regen_prompt", "phase4_regeneration")
graph.add_edge("phase4_regeneration", "phase5_finalize")
graph.add_edge("phase5_finalize", END)

# LangGraph Studio에서 인식할 엔트리포인트
app = graph.compile()
