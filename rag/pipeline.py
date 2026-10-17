# 검색 결과를 근거로 LLM이 답변을 생성하는 파일. RAG 파이프라인의 최종 조립 지점.
from common.ai_model import get_llm_model
from rag.reranker import rerank
from rag.retriever import hybrid_search, metadata_filter_search, shares_vocabulary

LAW_NAME = "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법"

# 1차 검색(하이브리드)이 "관련 없음"으로 판단했을 때만, 후보군을 넓혀서 리랭킹으로 재시도한다
# (20개로는 RRF 융합 과정에서 정답이 후보군 밖으로 밀려나는 경우가 있어 30으로 설정)
RERANK_CANDIDATE_K = 30

ANSWER_PROMPT = """다음은 "{law_name}"의 법률 조문입니다. 이 내용만 근거로 삼아 질문에 답변하세요.
조문에 없는 내용은 답변하지 말고, 모른다고 답하세요.

[법률 조문]
{context}

[질문]
{query}

답변을 작성한 다음, 맨 마지막 줄에 아래 둘 중 하나만 정확히 출력하세요.
- 위 조문 내용으로 질문에 실제로 답할 수 있었다면: RELEVANT
- 질문이 조문 내용과 관련이 없거나 답할 수 없었다면: NOT_RELEVANT

[답변]"""


def _format_article_label(chunk):
    # 출처 표시용 레이블을 만든다 (예: "제6조(인공지능 기본계획의 수립) ①항")
    label = f"{chunk['article_label']}({chunk['title']})"
    if chunk["paragraph_no"]:
        label += f" {chunk['paragraph_no']}항"
    if chunk["item_no"]:
        label += f" {chunk['item_no']}"
    return label


def _build_context(chunks):
    # 검색된 청크들을 LLM에게 보여줄 하나의 텍스트로 합친다
    return "\n\n".join(
        f"[{_format_article_label(chunk)}]\n{chunk['content']}" for chunk in chunks
    )


def _split_relevance_flag(text):
    # 답변 맨 마지막 줄의 RELEVANT/NOT_RELEVANT 표시를 떼어내고, (답변 본문, 관련 여부)를 반환한다
    lines = text.strip().splitlines()
    last_line = lines[-1].strip() if lines else ""

    if last_line == "NOT_RELEVANT":
        return "\n".join(lines[:-1]).strip(), False
    if last_line == "RELEVANT":
        return "\n".join(lines[:-1]).strip(), True

    # LLM이 표시를 안 붙였으면, 안전하게 "관련 있음"으로 취급한다
    return text.strip(), True


def _generate_answer(query, chunks):
    # 검색된 조문을 근거(context)로 LLM 답변을 생성하고, (답변, 관련 여부)를 반환한다
    llm = get_llm_model()
    prompt = ANSWER_PROMPT.format(
        law_name=LAW_NAME,
        context=_build_context(chunks),
        query=query,
    )
    response = llm.invoke(prompt)
    return _split_relevance_flag(response.content)


def answer_question(query, top_k=5):
    # 1. 조문번호가 언급되면 메타데이터 필터링으로, 아니면 하이브리드 검색(Dense+BM25+RRF)으로 top_k를 가져온다
    top_chunks = metadata_filter_search(query, top_k=top_k)
    answer, grounded = _generate_answer(query, top_chunks)

    # 2. 1차 검색으로 "관련 없음" 판정이 나온 경우에만, 후보군을 넓혀 리랭킹으로 한 번 더 시도한다
    #    (평균 성능을 깎아먹는 리랭킹을 상시 적용하지 않으면서도, 하이브리드 검색이 놓친 케이스를 구제하기 위함)
    #    단, 질문이 법 내용과 어휘조차 안 겹치면(예: "배고파") 재시도해도 구제될 가능성이 없으므로
    #    비용이 큰 리랭킹을 돌리지 않고 바로 포기한다
    if not grounded and shares_vocabulary(query):
        candidates = hybrid_search(query, top_k=RERANK_CANDIDATE_K, candidate_k=RERANK_CANDIDATE_K)
        reranked_chunks = rerank(query, candidates, top_k=top_k)
        if reranked_chunks:
            retry_answer, retry_grounded = _generate_answer(query, reranked_chunks)
            if retry_grounded:
                top_chunks, answer, grounded = reranked_chunks, retry_answer, retry_grounded

    # 3. 답변, 관련 여부, 근거 조문을 함께 반환한다
    return {
        "answer": answer,
        "grounded": grounded,
        "sources": [
            {"article": _format_article_label(chunk), "content": chunk["content"]}
            for chunk in top_chunks
        ],
    }
