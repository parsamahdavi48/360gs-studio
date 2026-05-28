from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_MAX_XML_BYTES = 256 * 1024 * 1024
XML_DECL_SCAN_BYTES = 1024 * 1024


def parse_xml_file(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_XML_BYTES,
) -> ET.ElementTree:
    xml_path = Path(path)
    size = xml_path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"XML file is too large: {xml_path} ({size} bytes, limit {max_bytes})")

    with xml_path.open("rb") as f:
        prefix = f.read(XML_DECL_SCAN_BYTES).upper()
    if b"<!DOCTYPE" in prefix or b"<!ENTITY" in prefix:
        raise ValueError(f"XML DOCTYPE and ENTITY declarations are not supported: {xml_path}")

    return ET.parse(xml_path)
