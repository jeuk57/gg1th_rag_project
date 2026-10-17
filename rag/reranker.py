# 검색 결과를 전용 cross-encoder 모델로 재정렬(리랭킹)하는 파일.
# 실험 결과 BM25를 형태소 분석(Kiwi)으로 고친 뒤로는 리랭킹을 상시 적용하면 오히려 평균
# 성능을 깎아먹어서, pipeline.py에서는 1차 답변이 "관련없음"으로 판정된 경우에만 선택적으로 호출한다.
from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

_model = None


def _get_model():
    # 모델은 처음 호출될 때 한 번만 다운로드/로딩한다 (이후엔 재사용)
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL_NAME)
    return _model


def score_candidates(query, chunks):
    # (질문, 청크) 쌍을 cross-encoder에 넣어 점수를 매기고, 점수 높은 순으로 (청크, 점수)를 반환한다
    model = _get_model()

    pairs = [[query, chunk["content"]] for chunk in chunks]
    scores = model.predict(pairs)

    return sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)


def rerank(query, chunks, top_k=5, score_threshold=None):
    # score_threshold를 넘는 점수가 하나도 없으면 "관련 문서 없음"으로 보고 빈 리스트를 반환한다
    ranked = score_candidates(query, chunks)

    if score_threshold is not None and (not ranked or ranked[0][1] < score_threshold):
        return []

    return [chunk for chunk, _ in ranked[:top_k]]
