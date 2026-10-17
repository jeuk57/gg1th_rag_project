# .env에서 환경변수를 읽어와 프로젝트 전체에서 쓰는 설정값으로 만드는 파일.
import os

from dotenv import load_dotenv

load_dotenv(override=True)

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")

MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")
TEMPERATURE = os.getenv("LLM_TEMPERATURE", 2)
MAX_TOKENS = os.getenv("LLM_MAX_TOKENS", 2086)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

LAW_API_OC = os.getenv("LAW_API_OC")
LAW_API_TARGET_MST = os.getenv("LAW_API_TARGET_MST")
