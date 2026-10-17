# 법제처 Open API로 법령 원문을 받아와서, 장/절/조/항/호/목 구조로 파싱하는 파일.
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional

from common.config import LAW_API_OC

LAW_API_BASE = "https://www.law.go.kr/DRF"

_CHAPTER_RE = re.compile(r"^제\d+장")
_SECTION_RE = re.compile(r"^제\d+절")


def _clean(text: Optional[str]) -> str:
    # 공백/줄바꿈을 하나의 띄어쓰기로 정리한다
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _request(path: str, params: dict) -> str:
    # 법령 API에 GET 요청을 보내고 응답 본문(XML)을 문자열로 반환한다
    query = urllib.parse.urlencode(params)
    url = f"{LAW_API_BASE}/{path}?{query}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.read().decode("utf-8")


def search_law(query: str, oc: str = LAW_API_OC, display: int = 20) -> list[dict]:
    """법령명으로 검색하여 법령일련번호(MST) 등 기본 정보를 반환한다."""
    xml_text = _request(
        "lawSearch.do",
        {"OC": oc, "target": "law", "type": "XML", "query": query, "display": display},
    )
    root = ET.fromstring(xml_text)
    return [
        {
            "law_id": law.findtext("법령ID", ""),
            "mst": law.findtext("법령일련번호", ""),
            "name": law.findtext("법령명한글", ""),
            "short_name": law.findtext("법령약칭명", ""),
            "promulgation_date": law.findtext("공포일자", ""),
            "enforcement_date": law.findtext("시행일자", ""),
            "ministry": law.findtext("소관부처명", ""),
        }
        for law in root.findall(".//law")
    ]


def fetch_law_xml(mst: str, oc: str = LAW_API_OC) -> str:
    """법령일련번호(MST)로 법령 본문 전체를 XML로 가져온다."""
    return _request("lawService.do", {"OC": oc, "target": "law", "type": "XML", "MST": mst})


def _parse_items(parent: ET.Element) -> list[dict]:
    """조문단위 또는 항 아래의 호(목 포함) 목록을 파싱한다."""
    items = []
    for ho in parent.findall("호"):
        sub_items = [
            {"no": _clean(mok.findtext("목번호")), "text": _clean(mok.findtext("목내용"))}
            for mok in ho.findall("목")
        ]
        items.append(
            {
                "no": _clean(ho.findtext("호번호")),
                "text": _clean(ho.findtext("호내용")),
                "sub_items": sub_items,
            }
        )
    return items


def _parse_paragraphs(unit: ET.Element) -> list[dict]:
    """조문단위 아래의 항(호/목 포함) 목록을 파싱한다."""
    paragraphs = [
        {
            "no": _clean(hang.findtext("항번호")),
            "text": _clean(hang.findtext("항내용")),
            "items": _parse_items(hang),
        }
        for hang in unit.findall("항")
    ]

    # 항 없이 조문단위 바로 아래 호가 오는 경우(예: 정의 조항)도 하나의 단락으로 취급한다.
    direct_items = _parse_items(unit)
    if direct_items:
        paragraphs.append({"no": None, "text": "", "items": direct_items})

    return paragraphs


def _parse_article(unit: ET.Element, chapter: Optional[str], section: Optional[str]) -> dict:
    # 조문단위 XML 하나를 파싱해서 조번호/제목/항 목록/전체 텍스트를 담은 딕셔너리로 만든다
    article_no = _clean(unit.findtext("조문번호"))
    branch_no = _clean(unit.findtext("조문가지번호")) or None
    label = f"제{article_no}조" + (f"의{branch_no}" if branch_no else "")

    paragraphs = _parse_paragraphs(unit)
    lead_text = _clean(unit.findtext("조문내용"))

    full_text_parts = [lead_text] if lead_text else []
    for para in paragraphs:
        if para["text"]:
            full_text_parts.append(para["text"])
        for item in para["items"]:
            full_text_parts.append(item["text"])
            for sub in item["sub_items"]:
                full_text_parts.append(sub["text"])

    return {
        "chapter": chapter,
        "section": section,
        "article_no": article_no,
        "branch_no": branch_no,
        "article_label": label,
        "title": _clean(unit.findtext("조문제목")),
        "paragraphs": paragraphs,
        "full_text": "\n".join(full_text_parts),
    }


def parse_law_xml(xml_text: str) -> dict:
    """법령 본문 XML을 장/절 메타데이터가 붙은 조문 리스트로 변환한다."""
    root = ET.fromstring(xml_text)

    # API 키/IP 오류 등으로 실패하면 <법령> 대신 <Response> 형태로 에러 메시지가 내려온다
    if root.tag != "법령":
        msg = root.findtext("msg") or root.findtext("result") or xml_text
        raise RuntimeError(f"법령 API 응답 오류: {msg}")

    info = root.find("기본정보")

    meta = {
        "law_id": info.findtext("법령ID", ""),
        "name": _clean(info.findtext("법령명_한글")),
        "short_name": _clean(info.findtext("법령명약칭")),
        "promulgation_date": info.findtext("공포일자", ""),
        "enforcement_date": info.findtext("시행일자", ""),
        "ministry": info.findtext("소관부처", ""),
    }

    articles = []
    chapter: Optional[str] = None
    section: Optional[str] = None
    for unit in root.findall(".//조문/조문단위"):
        if unit.findtext("조문여부") == "전문":
            heading = _clean(unit.findtext("조문내용"))
            if _CHAPTER_RE.match(heading):
                chapter, section = heading, None
            elif _SECTION_RE.match(heading):
                section = heading
            continue
        articles.append(_parse_article(unit, chapter, section))

    supplementary = [
        {
            "promulgation_date": unit.findtext("부칙공포일자", ""),
            "promulgation_no": unit.findtext("부칙공포번호", ""),
            "text": _clean(unit.findtext("부칙내용")),
        }
        for unit in root.findall(".//부칙/부칙단위")
    ]

    return {"meta": meta, "articles": articles, "supplementary": supplementary}


def load_law(mst: str, oc: str = LAW_API_OC) -> dict:
    """법령일련번호(MST)로 조문 구조까지 파싱된 법령 데이터를 반환한다 (API 호출)."""
    return parse_law_xml(fetch_law_xml(mst, oc=oc))


def load_law_from_file(path: str) -> dict:
    """law.go.kr API 원문을 저장해둔 로컬 XML 파일(data/ai_basic_law/law_raw.xml)을
    읽어서 조문 구조까지 파싱된 법령 데이터를 반환한다 (API 호출 없음)."""
    with open(path, encoding="utf-8") as f:
        xml_text = f.read()
    return parse_law_xml(xml_text)
