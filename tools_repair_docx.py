from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.etree import ElementTree as ET


SRC = Path(r"C:\Users\hasee\Desktop\paper.docx")
OUT = Path(r"D:\pycharm\rgcl_llm\paper_revision_work\paper_repaired_working.docx")

REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

ET.register_namespace("", REL_NS)
ET.register_namespace("w", W_NS)
ET.register_namespace("r", R_NS)


def remove_null_image_relationship(data: bytes) -> bytes:
    root = ET.fromstring(data)
    removed_ids = []
    for rel in list(root):
        if rel.attrib.get("Target") == "../NULL":
            removed_ids.append(rel.attrib.get("Id"))
            root.remove(rel)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), removed_ids


def remove_broken_drawings(data: bytes, rel_ids: list[str]) -> bytes:
    if not rel_ids:
        return data
    ns = {
        "w": W_NS,
        "r": R_NS,
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    }
    root = ET.fromstring(data)
    parent = {}
    for p in root.iter():
        for c in p:
            parent[c] = p
    for blip in list(root.findall(".//a:blip", ns)):
        embed = blip.attrib.get(f"{{{R_NS}}}embed")
        if embed in rel_ids:
            drawing = blip
            while drawing is not None and drawing.tag != f"{{{W_NS}}}drawing":
                drawing = parent.get(drawing)
            if drawing is not None:
                run = parent.get(drawing)
                if run is not None:
                    run.remove(drawing)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main():
    OUT.parent.mkdir(exist_ok=True)
    removed_ids = []
    with ZipFile(SRC, "r") as zin, ZipFile(OUT, "w", ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/_rels/document.xml.rels":
                data, removed_ids = remove_null_image_relationship(data)
            elif item.filename == "word/document.xml":
                data = remove_broken_drawings(data, removed_ids)
            zout.writestr(item, data)
    print(f"wrote {OUT}")
    print("removed relationships:", removed_ids)


if __name__ == "__main__":
    main()
