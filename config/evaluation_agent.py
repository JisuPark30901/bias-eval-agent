# evaluation_agent.py
import pandas as pd
import json
import os
from openai import OpenAI
from evaluation_config import EVALUATION_RUBRICS, AGENT_PROMPT_TEMPLATE

# OpenAI 클라이언트 초기화 (API 키는 환경 변수에서 로드)
client = OpenAI()

def get_llm_evaluation(prompt):
    """LLM을 호출하여 평가 점수와 이유를 받아오는 함수"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 고성능 모델 사용 권장
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, # 일관된 평가를 위해 0으로 설정
            response_format={"type": "json_object"} # JSON 출력 강제
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error during LLM evaluation: {e}")
        return {"score": 0, "reasoning": "Error occurred"}

def evaluate_single_dialogue(row, dialogue_content, eval_type, result_json):
    """개별 대화에 대해 특정 에이전트가 평가 수행"""
    rubric_config = EVALUATION_RUBRICS[eval_type]
    
    # 프롬프트 구성
    prompt = AGENT_PROMPT_TEMPLATE.format(
        prompt=row["prompt"],
        problematic_sentence=result_json["problematic_sentence"],
        topic=result_json["topic"],
        problem=result_json["problem"],
        similar_dialogues=dialogue_content,
        evaluation_name=rubric_config["name"],
        evaluation_description=rubric_config["description"],
        evaluation_rubric=rubric_config["rubric"]
    )
    
    # LLM 호출 및 결과 반환
    return get_llm_evaluation(prompt)

def run_evaluation(input_file_path, output_file_path):
    """전체 데이터셋에 대해 평가 실행"""
    xls = pd.ExcelFile(input_file_path)
    all_results = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        for index, row in df.iterrows():
            try:
                # 결과 JSON 파싱 (마크다운 코드 블록 처리 포함)
                result_str = str(row["result"]).replace("```json", "").replace("```", "").strip()
                result_json = json.loads(result_str)
                
                # 생성된 대화 추출
                dialogues = [result_json.get(f"similar_problematic_dialogue_{i}", "") for i in range(1, 6)]
                
                # 1. 개별 대화 평가 (Topic, Realism, Context, Intensity)
                for i, dialogue in enumerate(dialogues, 1):
                    if not dialogue: continue
                    
                    for eval_type in ["topic_consistency", "realism", "context_consistency", "intensity"]:
                        eval_result = evaluate_single_dialogue(row, dialogue, eval_type, result_json)
                        
                        all_results.append({
                            "sheet": sheet_name,
                            "qid": row["qid"],
                            "dialogue_index": i,
                            "evaluation_type": eval_type,
                            "agent_name": EVALUATION_RUBRICS[eval_type]["name"],
                            "score": eval_result["score"],
                            "reasoning": eval_result["reasoning"]
                        })

                # 2. 전체 세트 평가 (Diversity)
                all_dialogues_str = "\n".join([f"{i}. {d}" for i, d in enumerate(dialogues, 1) if d])
                if all_dialogues_str:
                    eval_result = evaluate_single_dialogue(row, all_dialogues_str, "diversity", result_json)
                    
                    all_results.append({
                        "sheet": sheet_name,
                        "qid": row["qid"],
                        "dialogue_index": "all",
                        "evaluation_type": "diversity",
                        "agent_name": EVALUATION_RUBRICS["diversity"]["name"],
                        "score": eval_result["score"],
                        "reasoning": eval_result["reasoning"]
                    })

            except Exception as e:
                print(f"Skipping row {index}: {e}")
                continue

    # 결과 저장
    pd.DataFrame(all_results).to_csv(output_file_path, index=False, encoding="utf-8-sig")
    print(f"Evaluation complete. Results saved to {output_file_path}")

if __name__ == "__main__":
    # 사용 예시
    run_evaluation("input_data.xlsx", "evaluation_results.csv")
