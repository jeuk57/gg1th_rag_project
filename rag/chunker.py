# loader.py가 파싱한 조문/부칙을 Qdrant에 저장할 청크(검색 단위)로 나누는 파일.
# 기본은 조 단위 하나가 청크 하나이고, 800자를 넘는 조문만 항/호 단위로 더 쪼갠다.
import re

from rag.loader import load_law, load_law_from_file

# 이 글자수를 넘는 조문만 항/호 단위로 더 쪼갠다
CHUNK_LENGTH_LIMIT = 800

# 부칙 내부의 "제N조(...)" 조문 경계, "다만," 예외 경계에서 나눈다
_SUPPLEMENTARY_SPLIT_RE = re.compile(r"(?=제\d+조(?:의\d+)?\()|(?= 다만,)")


def _merge_item_text(item):
    # 호(1. 2. 3.) 내용 + 그 아래 목(가. 나. 다.) 내용을 하나로 합친다
    texts = [item["text"]]
    for sub in item["sub_items"]:
        texts.append(sub["text"])
    return "\n".join(texts)


def _make_chunk(article, content, paragraph_no=None, item_no=None, add_prefix=True):
    # 청크 하나를 딕셔너리로 만든다
    chunk_id = article["article_label"]
    if paragraph_no:
        chunk_id += f"-{paragraph_no}"
    if item_no:
        chunk_id += f"-{item_no}"

    if add_prefix:
        # 항/호만 떼어낸 내용은 "몇 조인지" 알 수 없으므로 앞에 조문 제목을 붙여준다
        prefix = f"{article['article_label']}({article['title']})"
        content = f"{prefix}\n{content}"

    return {
        "chunk_id": chunk_id,
        "chapter": article["chapter"],
        "section": article["section"],
        "article_label": article["article_label"],
        "title": article["title"],
        "paragraph_no": paragraph_no,
        "item_no": item_no,
        "content": content,
    }


def split_article(article):
    # 조문 하나를 받아서 청크 리스트(1개 또는 여러 개)로 만든다
    full_text = article["full_text"]

    # 짧은 조문은 통째로 하나의 청크로 사용 (full_text에 이미 "제N조(제목)"이 포함돼 있음)
    if len(full_text) <= CHUNK_LENGTH_LIMIT:
        return [_make_chunk(article, full_text, add_prefix=False)]

    chunks = []
    for paragraph in article["paragraphs"]:
        if paragraph["no"]:
            # 항(①②③)이 실제로 있는 경우 -> 항 단위로 쪼갠다
            texts = [paragraph["text"]]
            for item in paragraph["items"]:
                texts.append(_merge_item_text(item))
            content = "\n".join(texts)
            chunks.append(_make_chunk(article, content, paragraph_no=paragraph["no"]))
        else:
            # 항 없이 바로 호(1. 2. 3.)가 오는 경우 -> 호 단위로 쪼갠다 (예: 제2조 정의)
            for item in paragraph["items"]:
                content = _merge_item_text(item)
                chunks.append(_make_chunk(article, content, item_no=item["no"]))

    return chunks


def _format_date(yyyymmdd):
    # "20260120" -> "2026.1.20"
    if not yyyymmdd or len(yyyymmdd) != 8:
        return yyyymmdd
    year, month, day = yyyymmdd[:4], int(yyyymmdd[4:6]), int(yyyymmdd[6:8])
    return f"{year}.{month}.{day}"


def split_supplementary(item):
    # 부칙 하나를 "제N조(...)" 경계, "다만," 예외 경계에서 나눈다
    # (짧아도 "시행일 문장 + 긴 예외 목록"이 한 청크에 섞이면 의미가 희석되는 문제가 있었음)
    label = f"부칙(제{item['promulgation_no']}호)"
    date = _format_date(item["promulgation_date"])
    header = f"{label}, {date} 공포"

    parts = [p.strip() for p in _SUPPLEMENTARY_SPLIT_RE.split(item["text"]) if p.strip()]
    if len(parts) <= 1:
        parts = [item["text"]]

    return [
        {
            "chunk_id": f"부칙-{item['promulgation_no']}-{i}",
            "chapter": "부칙",
            "section": None,
            "article_label": label,
            "title": f"{date} 공포",
            "paragraph_no": None,
            "item_no": str(i),
            "content": f"{header}\n{part}",
        }
        for i, part in enumerate(parts, start=1)
    ]


def _build_chunks_from_law(law):
    # 파싱된 법령 데이터(조문+부칙)로 전체 청크 리스트를 만든다
    law_name = law["meta"]["name"]

    all_chunks = []
    for article in law["articles"]:
        for chunk in split_article(article):
            chunk["law_name"] = law_name
            all_chunks.append(chunk)

    for item in law["supplementary"]:
        for chunk in split_supplementary(item):
            chunk["law_name"] = law_name
            all_chunks.append(chunk)

    return all_chunks


def build_chunks(mst):
    # 법제처 API로 법령을 받아와서 전체 청크 리스트를 만든다
    return _build_chunks_from_law(load_law(mst))


def build_chunks_from_file(path):
    # 로컬 XML 파일(law.go.kr 원문)로부터 전체 청크 리스트를 만든다 (API 호출 없음)
    return _build_chunks_from_law(load_law_from_file(path))
