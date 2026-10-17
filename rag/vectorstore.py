# 청크를 임베딩 벡터로 변환해서 Qdrant에 저장하는 파일.
from qdrant_client.models import Distance, PointStruct, VectorParams

from common.ai_model import get_embedding_model
from common.qdrant import get_qdrant_client
from common.config import LAW_API_TARGET_MST
from rag.chunker import build_chunks, build_chunks_from_file

COLLECTION_NAME = "law_articles"


def save_to_qdrant(mst=LAW_API_TARGET_MST):
    # 법제처 API로 법령을 받아와서 인덱싱한다 (LAW_API_OC/LAW_API_TARGET_MST 필요)
    _index_chunks(build_chunks(mst))


def save_to_qdrant_from_file(path):
    # 로컬 XML 파일(law.go.kr 원문)로부터 인덱싱한다 (API 호출 없음)
    _index_chunks(build_chunks_from_file(path))


def _index_chunks(chunks):
    # 1. Qdrant, Embedding 클라이언트 준비
    qdrant_client = get_qdrant_client()
    embedding_client = get_embedding_model()

    # 2. 청크 내용(content)을 임베딩 벡터로 변환
    texts = [chunk["content"] for chunk in chunks]
    vectors = embedding_client.embed_documents(texts)
    print(f"청크 개수: {len(vectors)}, 임베딩 차원: {len(vectors[0])}")

    # 3. 컬렉션을 항상 새로 만든다 (청크 내용이 바뀌어도 항상 최신 상태로 유지하기 위함)
    if qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.delete_collection(COLLECTION_NAME)

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
    )

    # 4. 청크 + 벡터를 포인트로 만들어서 Qdrant에 저장
    #    청크 딕셔너리를 그대로 payload로 사용 (chapter, article_label, content 등 전부 포함)
    points = [
        PointStruct(id=i, vector=vector, payload=chunk)
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"{len(points)}개 청크를 '{COLLECTION_NAME}' 컬렉션에 저장했습니다.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        save_to_qdrant_from_file(sys.argv[1])
    else:
        save_to_qdrant()
