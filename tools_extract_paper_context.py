from pathlib import Path
import json

from zipfile import ZipFile
from xml.etree import ElementTree as ET
import pdfplumber


ROOT = Path(r"D:\pycharm\rgcl_llm")
SOURCES = {
    "paper": Path(r"C:\Users\hasee\Desktop\paper.docx"),
    "electronics": Path(r"C:\Users\hasee\Desktop\electronics-14-03504-v2 (2).pdf"),
    "baseline": Path(r"C:\Users\hasee\Desktop\2024.acl-long.291.pdf"),
}
OUT = ROOT / "paper_revision_work"
OUT.mkdir(exist_ok=True)


def extract_docx(path: Path):
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    paragraphs = []
    for i, p in enumerate(root.findall(".//w:p", ns)):
        texts = [t.text or "" for t in p.findall(".//w:t", ns)]
        text = "".join(texts).strip()
        if text:
            pstyle = ""
            ppr = p.find("w:pPr", ns)
            if ppr is not None:
                st = ppr.find("w:pStyle", ns)
                if st is not None:
                    pstyle = st.attrib.get(f"{{{ns['w']}}}val", "")
            paragraphs.append({
                "index": i,
                "style": pstyle,
                "text": text,
            })
    tables = []
    for ti, table in enumerate(root.findall(".//w:tbl", ns)):
        rows = []
        for row in table.findall(".//w:tr", ns):
            cells = []
            for cell in row.findall("w:tc", ns):
                texts = [t.text or "" for t in cell.findall(".//w:t", ns)]
                cells.append("".join(texts).strip())
            rows.append(cells)
        tables.append({"index": ti, "rows": rows})
    return {"paragraphs": paragraphs, "tables": tables}


def extract_pdf(path: Path, max_pages=None):
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            if max_pages is not None and i >= max_pages:
                break
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            pages.append({"page": i + 1, "text": text})
    return {"pages": pages}


def main():
    paper = extract_docx(SOURCES["paper"])
    (OUT / "paper_extracted.json").write_text(
        json.dumps(paper, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "paper_text.txt").write_text(
        "\n\n".join(f"[{p['index']}|{p['style']}]\n{p['text']}" for p in paper["paragraphs"]),
        encoding="utf-8",
    )
    for key in ["electronics", "baseline"]:
        data = extract_pdf(SOURCES[key])
        (OUT / f"{key}_extracted.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (OUT / f"{key}_text.txt").write_text(
            "\n\n".join(f"--- Page {p['page']} ---\n{p['text']}" for p in data["pages"]),
            encoding="utf-8",
        )
    print(f"Wrote extraction outputs to {OUT}")


if __name__ == "__main__":
    main()
