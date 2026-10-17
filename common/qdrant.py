# Qdrant 클라이언트를 만들어주는 파일.
from qdrant_client import QdrantClient

from common.config import QDRANT_URL

def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL
    )