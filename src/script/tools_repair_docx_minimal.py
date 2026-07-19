from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import re


SRC = Path(r"C:\Users\hasee\Desktop\paper.docx")
OUT = Path(r"D:\pycharm\rgcl_llm\paper_revision_work\paper_repaired_working.docx")


def main():
    OUT.parent.mkdir(exist_ok=True)
    with ZipFile(SRC, "r") as zin, ZipFile(OUT, "w", ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/_rels/document.xml.rels":
                text = data.decode("utf-8", errors="strict")
                text = re.sub(
                    r'<Relationship\b[^>]*Id="rId10"[^>]*Target="\.\./NULL"[^>]*/>',
                    "",
                    text,
                )
                data = text.encode("utf-8")
            elif item.filename == "word/document.xml":
                text = data.decode("utf-8", errors="strict")
                text = re.sub(
                    r"<w:r\b(?:(?!</w:r>).)*?r:embed=\"rId10\"(?:(?!</w:r>).)*?</w:r>",
                    "",
                    text,
                    flags=re.S,
                )
                data = text.encode("utf-8")
            zout.writestr(item, data)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
