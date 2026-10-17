# 검색(Retrieval) 담당 파일. 질문을 받아 Qdrant에서 관련 조문 청크를 찾아온다.
# Dense(임베딩 유사도) + BM25(키워드) 검색을 RRF로 합친 하이브리드 검색이 기본이고,
# 질문에 조문번호가 직접 적혀 있으면 임베딩 없이 바로 그 조문만 가져오는 메타데이터 필터링도 있다.
import re

from kiwipiepy import Kiwi
from qdrant_client.models import FieldCondition, Filter, MatchValue
from rank_bm25 import BM25Okapi

from common.ai_model import get_embedding_model
from common.qdrant import get_qdrant_client
from rag.vectorstore import COLLECTION_NAME

RRF_K = 60  # RRF 공식에서 관례적으로 쓰는 상수

_kiwi = Kiwi()
_CONTENT_POS_PREFIXES = ("N", "V", "XR", "SL", "SN")  # 명사/동사·형용사/어근/외국어/숫자만 남김 (조사·어미 제외)

_corpus_vocabulary = None  # shares_vocabulary()에서 한 번만 계산해서 재사용

_ARTICLE_RE = re.compile(r"제(\d+)조(?:의(\d+))?")


def _load_all_chunks():
    # Qdrant에 저장된 모든 청크(payload)를 가져온다
    qdrant_client = get_qdrant_client()
    points, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME, limit=1000, with_payload=True
    )
    return [{"id": point.id, "payload": point.payload} for point in points]


def _dense_search(query, top_k):
    # 질문을 임베딩해서 Qdrant에서 가장 비슷한 청크 id 순서를 찾는다
    embedding_client = get_embedding_model()
    qdrant_client = get_qdrant_client()

    query_vector = embedding_client.embed_query(query)
    result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )
    return [point.id for point in result.points]


def _tokenize(text):
    # 형태소 분석으로 쪼갠 다음, 조사/어미 같은 의미 없는 토큰은 버리고 명사/동사 등만 남긴다
    return [token.form for token in _kiwi.tokenize(text) if token.tag.startswith(_CONTENT_POS_PREFIXES)]


def _bm25_search(query, chunks, top_k):
    # 전체 청크 내용으로 BM25 점수를 계산해서 가장 높은 것부터 순위를 매긴다
    tokenized_corpus = [_tokenize(chunk["payload"]["content"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk["id"] for chunk, _ in ranked[:top_k]]


def _reciprocal_rank_fusion(rank_lists):
    # 여러 검색 결과(순위 리스트)를 RRF 공식으로 하나의 점수로 합친다
    scores = {}
    for ranked_ids in rank_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (RRF_K + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query, top_k=5, candidate_k=20):
    # Dense 검색과 BM25 검색을 각각 넉넉하게 수행한 뒤 RRF로 합쳐서 top_k를 반환한다
    chunks = _load_all_chunks()
    dense_ids = _dense_search(query, candidate_k)
    bm25_ids = _bm25_search(query, chunks, candidate_k)

    fused = _reciprocal_rank_fusion([dense_ids, bm25_ids])
    top_ids = [chunk_id for chunk_id, _ in fused[:top_k]]

    chunk_by_id = {chunk["id"]: chunk["payload"] for chunk in chunks}
    return [chunk_by_id[chunk_id] for chunk_id in top_ids]


def _extract_article_label(query):
    # 질문에서 "제33조", "제17조의2" 같은 조문번호를 뽑아낸다 (없으면 None)
    match = _ARTICLE_RE.search(query)
    if not match:
        return None
    num, branch = match.groups()
    return f"제{num}조" + (f"의{branch}" if branch else "")


def shares_vocabulary(query):
    # 질문 토큰이 전체 청크 어휘와 하나도 안 겹치면, 법 내용과 아예 무관한 질문일 가능성이 높다
    # (리랭킹처럼 비용이 큰 재시도를 돌릴지 말지 빠르게 거르는 용도)
    global _corpus_vocabulary
    if _corpus_vocabulary is None:
        chunks = _load_all_chunks()
        _corpus_vocabulary = set()
        for chunk in chunks:
            _corpus_vocabulary.update(_tokenize(chunk["payload"]["content"]))

    return bool(set(_tokenize(query)) & _corpus_vocabulary)


def metadata_filter_search(query, top_k=5, candidate_k=20):
    # 질문에 조문번호가 직접 언급되면, 임베딩 없이 그 조문만 바로 필터링해서 가져온다
    # (언급이 없거나 존재하지 않는 조문번호면 하이브리드 검색으로 대체한다)
    label = _extract_article_label(query)
    if not label:
        return hybrid_search(query, top_k=top_k, candidate_k=candidate_k)

    qdrant_client = get_qdrant_client()
    points, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(key="article_label", match=MatchValue(value=label))]),
        limit=top_k,
    )
    chunks = [point.payload for point in points]

    if not chunks:
        return hybrid_search(query, top_k=top_k, candidate_k=candidate_k)
    return chunks
