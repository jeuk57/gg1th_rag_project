# AI 기본법 QA (RAG)

"인공지능 발전과 신뢰 기반 조성 등에 관한 기본법" 조문을 근거로 질문에 답하는 RAG 기반 QA
백엔드. 법제처 Open API로 법령 원문을 받아와 구조 단위(장/절/조/항/호/목)로 파싱·청킹하고,
Qdrant에 저장한 뒤 Dense+BM25 하이브리드 검색과 선택적 리랭킹으로 조문을 찾아 LLM이
그 조문만 근거로 답변을 생성한다.

## 아키텍처

```
법제처 Open API
     │ (rag/loader.py)
     ▼
장/절/조/항/호/목 구조로 파싱
     │ (rag/chunker.py)
     ▼
청크(800자 기준으로 조/항/호 단위 분리)
     │ (rag/vectorstore.py)
     ▼
Qdrant (벡터 + payload)
     │
     ▼
질문 ──▶ 메타데이터 필터링(조문번호 명시 질문) ──▶ 바로 조회
     │                                        (rag/retriever.py)
     └──▶ Hybrid 검색 (Dense + Kiwi 형태소 BM25 + RRF)
                   │
                   ▼
          LLM 답변 생성 + 관련성 자가판단(RELEVANT/NOT_RELEVANT)
                   │ (rag/pipeline.py)
                   ▼
          "관련없음" 판정 시에만: 후보군 확장 + cross-encoder 리랭킹 재시도
                   │
                   ▼
              답변 + 근거 조문 반환 ──▶ FastAPI ──▶ 웹 UI
```

## 폴더 구조

```
common/        .env 설정, LLM/임베딩 클라이언트, Qdrant 클라이언트
rag/           loader → chunker → vectorstore → retriever → pipeline
app/           FastAPI 서버(main.py) + 정적 웹 UI(static/)
eval/          골든셋(쉬운셋 49 / 하드셋 13) 기반 검색·답변 품질 평가
scripts/       법령 원문을 받아 data/에 저장하는 스크립트
data/          법령 원문 XML (출처: 법제처 국가법령정보센터 Open API, law.go.kr)
docker-qdrant/ Qdrant 로컬 실행용 docker-compose
```

`data/ai_basic_law/law_raw.xml`은 법제처 국가법령정보센터 Open API(law.go.kr)로 받은
"인공지능 발전과 신뢰 기반 조성 등에 관한 기본법" 원문이다. 이 파일만 있으면 법제처 API 키
없이도 바로 청킹·인덱싱을 돌려볼 수 있다 (아래 실행 방법 4번 참고).

## 설계 결정 (실측 기반)

이론적으로 더 나아 보이는 방법이 실제로는 역효과를 낸 경우가 많아서, 매 단계를 직접
측정해서 결정했다.

- **청킹**: 800자 넘는 조문만 항/호 단위로 재귀적으로 더 쪼개봤더니, 핵심 키워드가 포함된
  상위 문맥이 사라져서 검색이 오히려 나빠졌다 → 1단계 분리만 채택.
- **BM25 토큰화**: 공백 분리 대신 Kiwi 형태소 분석(명사/동사 등 내용어만 추출)으로 바꾸자
  검색 성능이 가장 크게 개선됐다 (쉬운셋 Hit@1 0.633 → 0.837).
- **리랭킹**: BM25를 고치기 전에는 cross-encoder 리랭킹이 확실히 도움이 됐지만, BM25를
  고친 뒤에는 평균 성능을 오히려 깎아먹었다 → 상시 적용은 제외하고, 1차 답변이
  "관련없음"으로 판정된 경우에만 후보군을 넓혀 재시도하는 선택적 폴백으로 변경.
  (질문이 법 내용과 어휘 자체가 안 겹치는 경우는 재시도해도 구제될 수 없으므로, 비용이 큰
  리랭킹을 돌리기 전에 어휘 중복 여부로 먼저 거른다.)

## 평가

- 검색(Hit@K/Recall@K/MRR): 정답 조문 라벨과 기계적으로 비교, LLM 미사용
- 답변 품질(정답성/관련성/근거충실성/할루시네이션): LLM-as-judge
- 최신 평가 결과는 [`eval/results.txt`](eval/results.txt) 참고

```bash
uv run python eval/evaluate.py
```

## 실행 방법

1. 환경 변수 설정 (`.env.example`을 `.env`로 복사 후 값 채우기)
   - `LLM_API_KEY`/`LLM_BASE_URL`: OpenAI 호환 API (필수)
   - `LAW_API_OC`/`LAW_API_TARGET_MST`: [open.law.go.kr](https://open.law.go.kr)에서
     OC 발급(IP 화이트리스트 등록 필요). 아래 4번에서 저장된 XML 파일로 인덱싱할 거면
     생략해도 된다 — 법령 원문을 새로 받아오고 싶을 때만 필요하다.
2. Qdrant 실행
   ```bash
   cd docker-qdrant && docker compose up -d
   ```
3. 의존성 설치
   ```bash
   uv sync
   ```
4. Qdrant에 인덱싱
   ```bash
   # data/ai_basic_law/law_raw.xml로 바로 인덱싱 (API 키 불필요)
   uv run python -m rag.vectorstore data/ai_basic_law/law_raw.xml

   # 또는 법제처 API로 직접 받아와서 인덱싱 (LAW_API_OC/LAW_API_TARGET_MST 필요)
   uv run python -m rag.vectorstore
   ```
5. 서버 실행
   ```bash
   uv run uvicorn app.main:app --reload --port 8000
   ```
   `http://localhost:8000`에서 웹 UI로 질문 가능.
