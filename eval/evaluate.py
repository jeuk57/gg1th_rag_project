# 검색/답변 품질을 평가하는 파일.
# 검색 평가(Hit@K, MRR)는 정답 조문 라벨과 기계적으로 비교하고, 답변 평가는 LLM이 채점한다.
import json
import re
import time

from common.ai_model import get_llm_model
from rag.pipeline import answer_question
from rag.retriever import metadata_filter_search

GOLDEN_SET_PATH = "eval/golden_set.jsonl"
NEGATIVE_SET_PATH = "eval/negative_set.jsonl"  # 법에 없는 내용을 묻는 질문 (정답 조문이 없어야 정상)
SLEEP_SECONDS = 4  # API 분당 요청 제한(30회)을 피하기 위해 질문마다 쉬어가는 시간
ANSWER_SLEEP_SECONDS = 10  # 답변 평가는 질문 하나당 API 호출이 많아서 더 길게 쉰다

JUDGE_PROMPT = """다음은 법률 QA 시스템이 생성한 답변을 평가하는 작업입니다.

[질문]
{query}

[검색된 근거 조문]
{context}

[생성된 답변]
{answer}

아래 4가지 기준으로 답변을 평가하세요.
1. 정답성: 질문에 대해 맞는 답을 했는가 (1~5점)
2. 관련성: 질문과 관련 있는 내용으로 답했는가 (1~5점)
3. 근거충실성: 답변 내용이 위 "검색된 근거 조문"에 실제로 있는 내용인가 (1~5점)
4. 할루시네이션: 근거 조문에 없는 내용을 지어냈는가 (지어냈으면 1, 아니면 0)

다른 설명 없이 아래 형식으로만 답하세요.
정답성,관련성,근거충실성,할루시네이션
예시: 5,5,4,0"""


def load_golden_set():
    # 질문 + 정답 조문 목록이 들어있는 골든셋을 읽어온다
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_negative_set():
    # 법에 없는 내용을 묻는 질문 목록을 읽어온다 (정답 조문이 없는 게 정상)
    with open(NEGATIVE_SET_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def article_labels(chunks):
    # 청크 리스트에서 조문 라벨만 중복 없이 순서대로 뽑는다 (예: "제2조-4." -> "제2조")
    labels = []
    for chunk in chunks:
        label = chunk["article_label"]
        if label not in labels:
            labels.append(label)
    return labels


def hit_at_k(retrieved_labels, answer_labels):
    # 정답 조문이 검색 결과 안에 하나라도 있으면 True
    return any(label in answer_labels for label in retrieved_labels)


def recall_at_k(retrieved_labels, answer_labels):
    # 정답 조문 중 몇 개를 검색 결과에서 찾았는지 비율로 반환
    found = sum(1 for label in answer_labels if label in retrieved_labels)
    return found / len(answer_labels)


def reciprocal_rank(retrieved_labels, answer_labels):
    # 정답 조문이 몇 등으로 나왔는지를 1/등수 점수로 반환 (못 찾으면 0)
    for rank, label in enumerate(retrieved_labels, start=1):
        if label in answer_labels:
            return 1 / rank
    return 0.0


def evaluate_retrieval(search_fn, top_k=5, sleep_seconds=None, golden_set_path=None):
    path = golden_set_path or GOLDEN_SET_PATH
    with open(path, encoding="utf-8") as f:
        golden_set = [json.loads(line) for line in f if line.strip()]
    sleep_seconds = SLEEP_SECONDS if sleep_seconds is None else sleep_seconds

    hit1, hit3, hit5, recalls, rr_list = [], [], [], [], []
    for item in golden_set:
        chunks = search_fn(item["question"], top_k=top_k)
        labels = article_labels(chunks)
        answers = item["answer_articles"]

        # top_k개를 한 번만 검색하고, 그 안에서 1위/3위/5위까지만 잘라서 각각 채점한다
        hit1.append(hit_at_k(labels[:1], answers))
        hit3.append(hit_at_k(labels[:3], answers))
        hit5.append(hit_at_k(labels[:5], answers))
        recalls.append(recall_at_k(labels, answers))
        rr_list.append(reciprocal_rank(labels, answers))

        print(f"  [{len(hit1)}/{len(golden_set)}] {item['question']}")
        time.sleep(sleep_seconds)

    return {
        "Hit@1": sum(hit1) / len(hit1),
        "Hit@3": sum(hit3) / len(hit3),
        "Hit@5": sum(hit5) / len(hit5),
        "Recall@5": sum(recalls) / len(recalls),
        "MRR": sum(rr_list) / len(rr_list),
    }


def _build_judge_context(sources):
    # pipeline.answer_question()이 반환한 sources(근거 조문)를 평가용 텍스트로 합친다
    return "\n\n".join(f"[{s['article']}]\n{s['content']}" for s in sources)


def judge_answer(query, answer, sources):
    # LLM에게 (질문, 근거, 답변)을 보여주고 4가지 기준으로 점수를 매기게 한다
    llm = get_llm_model()
    prompt = JUDGE_PROMPT.format(
        query=query,
        context=_build_judge_context(sources),
        answer=answer,
    )
    response = llm.invoke(prompt)

    numbers = [int(n) for n in re.findall(r"\d+", response.content)]
    correctness, relevance, faithfulness, hallucination = numbers[:4]

    return {
        "correctness": correctness,
        "relevance": relevance,
        "faithfulness": faithfulness,
        "hallucination": hallucination,
    }


def evaluate_answers(golden_set_path=None):
    path = golden_set_path or GOLDEN_SET_PATH
    with open(path, encoding="utf-8") as f:
        golden_set = [json.loads(line) for line in f if line.strip()]

    scores = []
    for item in golden_set:
        result = answer_question(item["question"])
        score = judge_answer(item["question"], result["answer"], result["sources"])
        scores.append(score)

        print(f"  [{len(scores)}/{len(golden_set)}] {item['question']} -> grounded={result['grounded']} {score}")
        time.sleep(ANSWER_SLEEP_SECONDS)

    n = len(scores)
    return {
        "정답성": sum(s["correctness"] for s in scores) / n,
        "관련성": sum(s["relevance"] for s in scores) / n,
        "근거충실성": sum(s["faithfulness"] for s in scores) / n,
        "할루시네이션 비율": sum(s["hallucination"] for s in scores) / n,
    }


def evaluate_negative():
    # 법에 없는 내용을 묻는 질문에 "모른다"고 정직하게 답하는지 확인한다
    negative_set = load_negative_set()

    results = []
    for item in negative_set:
        result = answer_question(item["question"])
        score = judge_answer(item["question"], result["answer"], result["sources"])
        results.append({"question": item["question"], "answer": result["answer"], "score": score})

        print(f"  Q: {item['question']}")
        print(f"  A: {result['answer']}")
        print(f"  채점: {score}")
        print()
        time.sleep(ANSWER_SLEEP_SECONDS)

    return results


if __name__ == "__main__":
    print("=== 검색 평가 (메타데이터 필터링 + 하이브리드) ===")
    print(evaluate_retrieval(metadata_filter_search))
    print()

    print("=== 답변 품질 평가 (LLM-as-judge) ===")
    print(evaluate_answers())
