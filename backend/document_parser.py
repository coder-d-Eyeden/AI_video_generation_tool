"""
Document and Grounding Image Parser.
Handles PDF, DOCX, TXT document extraction and grounding photo validation.
Strictly requires real documents and images without dummy fallback.
"""
import io
import base64
from typing import List, Tuple
from pypdf import PdfReader
import docx
from PIL import Image


class DocumentParsingError(Exception):
    """Raised when uploaded file cannot be parsed or is empty."""
    pass


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from uploaded PDF, DOCX, or TXT document."""
    if not file_bytes or len(file_bytes.strip()) == 0:
        raise DocumentParsingError("The uploaded document is completely empty.")

    lower_name = filename.lower()
    text = ""

    try:
        if lower_name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(file_bytes))
            if len(reader.pages) == 0:
                raise DocumentParsingError("The PDF document contains 0 pages.")
            extracted_pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                extracted_pages.append(f"--- Page {i+1} ---\n{page_text}")
            text = "\n".join(extracted_pages).strip()

        elif lower_name.endswith(".docx") or lower_name.endswith(".doc"):
            doc = docx.Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_data = [c.text.strip() for c in row.cells if c.text.strip()]
                    if row_data:
                        paragraphs.append(" | ".join(row_data))
            text = "\n".join(paragraphs).strip()

        elif lower_name.endswith(".txt") or lower_name.endswith(".md"):
            try:
                text = file_bytes.decode("utf-8").strip()
            except UnicodeDecodeError:
                text = file_bytes.decode("latin-1", errors="ignore").strip()

        else:
            # Attempt plain text decoding as fallback
            try:
                text = file_bytes.decode("utf-8").strip()
            except UnicodeDecodeError:
                raise DocumentParsingError(
                    f"Unsupported file format for '{filename}'. Please upload a PDF (.pdf), Word document (.docx), or Text file (.txt)."
                )

    except Exception as e:
        if isinstance(e, DocumentParsingError):
            raise
        raise DocumentParsingError(f"Failed to read file '{filename}': {str(e)}")

    import re
    # Check if we extracted actual readable words, not just our injected "--- Page N ---" markers
    alpha_chars = re.sub(r'[^a-zA-Z]', '', text)
    if len(alpha_chars) < 50:
        raise DocumentParsingError(
            f"The uploaded document '{filename}' has a complex font encoding that could not be read automatically (very common with Indian Government PDFs). Please copy the text from the PDF and paste it into the 'Paste Document Text' tab instead."
        )

    return text


def process_grounding_images(images_data: List[Tuple[str, bytes]]) -> List[dict]:
    """
    Validates and encodes local grounding photos (college, libraries, campuses).
    Returns list of dicts with filename, mime_type, and base64 string.
    """
    valid_images = []
    for filename, data in images_data:
        if not data or len(data) == 0:
            continue
        try:
            pil_img = Image.open(io.BytesIO(data))
            img_format = (pil_img.format or "JPEG").lower()
            if img_format == "jpeg":
                mime_type = "image/jpeg"
            elif img_format == "png":
                mime_type = "image/png"
            elif img_format == "webp":
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"

            b64_str = base64.b64encode(data).decode("utf-8")
            valid_images.append({
                "filename": filename,
                "mime_type": mime_type,
                "base64": b64_str,
                "width": pil_img.width,
                "height": pil_img.height,
            })
        except Exception as e:
            raise DocumentParsingError(f"Invalid image file '{filename}': {str(e)}")

    return valid_images
