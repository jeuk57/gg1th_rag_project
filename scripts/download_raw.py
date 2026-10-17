# 법령 원문을 API로 받아서 data/ai_basic_law/에 파일로 저장하는 스크립트.
# 구조 분석(직접 눈으로 보고 청킹 방식을 정하는 것)을 위해 한 번 받아두는 용도다.
import os
import xml.dom.minidom

from common.config import LAW_API_TARGET_MST
from rag.loader import fetch_law_xml

OUTPUT_DIR = "data/ai_basic_law"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. API가 주는 원문 그대로 저장 (가공 없음)
    raw_xml = fetch_law_xml(LAW_API_TARGET_MST)
    raw_path = os.path.join(OUTPUT_DIR, "law_raw.xml")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_xml)
    print(f"원본 XML 저장: {raw_path}")

    # 2. 들여쓰기만 정리한 버전 (내용은 동일, 눈으로 읽기 편하게)
    pretty_xml = xml.dom.minidom.parseString(raw_xml).toprettyxml(indent="  ")
    pretty_path = os.path.join(OUTPUT_DIR, "law_pretty.xml")
    with open(pretty_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)
    print(f"읽기 쉬운 버전 저장: {pretty_path}")


if __name__ == "__main__":
    main()
