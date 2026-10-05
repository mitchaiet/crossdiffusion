"""Validate uploads and render selected PDF pages for the puzzle reader.

PDFium is not thread-safe, even across separate documents. Keep its handles and
calls inside the shared lock; return only encoded images and plain text.
"""

from __future__ import annotations

import base64
import io
import math
import re
import threading
from dataclasses import dataclass, field

import pypdfium2 as pdfium
from PIL import Image, UnidentifiedImageError

MAX_UPLOAD_BYTES = 20_000_000
MAX_IMAGE_BYTES = 12_000_000
MAX_PDF_PAGES = 20
MAX_SELECTED_PAGES = 6
MAX_PAGE_SIDE = 2200
PREVIEW_SIDE = 640
MAX_DOCUMENT_TEXT = 30_000
_PDF_LOCK = threading.Lock()


class DocumentError(ValueError):
    def __init__(self, detail: str, status_code: int = 422):
        super().__init__(detail)
        self.status_code = status_code


@dataclass
class ReaderDocument:
    image: bytes
    additional_images: list[bytes] = field(default_factory=list)
    text: str | None = None
    kind: str = "image"

    def reader_options(self) -> dict:
        if self.kind == "image":
            return {}
        return {"additional_images": self.additional_images, "document_text": self.text}


def upload_kind(data: bytes, name: str = "", content_type: str = "") -> str:
    if len(data) > MAX_UPLOAD_BYTES:
        raise DocumentError("File exceeds the 20 MB upload limit.", 413)
    if not data:
        raise DocumentError("The uploaded file is empty.")
    # PDF headers may occur after a short preamble. Check bytes, not just the
    # browser's MIME type: PDFs also arrive as application/octet-stream.
    if re.search(rb"%PDF-\d\.\d", data[:1024]):
        return "pdf"
    if name.lower().endswith(".pdf") or content_type.split(";")[0] == "application/pdf":
        raise DocumentError("This file does not contain a valid PDF header.")
    if len(data) > MAX_IMAGE_BYTES:
        raise DocumentError("Image exceeds the 12 MB upload limit.", 413)
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > 30_000_000:
                raise DocumentError("Image is too large to read; resize it below 30 megapixels.", 413)
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        if isinstance(error, DocumentError):
            raise
        raise DocumentError("Upload a PDF or a readable image such as PNG or JPEG.", 415) from error
    except Image.DecompressionBombError as error:
        raise DocumentError("Image dimensions are too large to read safely.", 413) from error
    return "image"


def selected_pages(value: str | None, page_count: int) -> list[int]:
    if value is None:
        return [1]
    parts = value.split(",")
    if len(value) > 100 or not value.strip() or any(not re.fullmatch(r"[0-9]+", part.strip()) for part in parts):
        raise DocumentError("Choose page numbers separated by commas, such as 1,2.")
    if len(parts) > MAX_SELECTED_PAGES:
        raise DocumentError(f"Choose at most {MAX_SELECTED_PAGES} pages for one puzzle.")
    pages = [int(part) for part in parts]
    if any(number < 1 or number > page_count for number in pages):
        raise DocumentError(f"Page numbers must be between 1 and {page_count}.")
    if len(pages) != len(set(pages)):
        raise DocumentError("Choose each PDF page only once.")
    return pages


def _page_count(pdf) -> int:
    count = len(pdf)
    if not count:
        raise DocumentError("The PDF has no pages.")
    if count > MAX_PDF_PAGES:
        raise DocumentError(f"PDF has {count} pages; upload a document with at most {MAX_PDF_PAGES} pages.")
    return count


def _render(page, max_side: int) -> bytes:
    width, height = page.get_size()
    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
        raise DocumentError("A PDF page has invalid dimensions.")
    scale = min(3.0, max_side / max(width, height))
    bitmap = page.render(scale=scale, may_draw_forms=False)
    try:
        image = bitmap.to_pil()
        try:
            output = io.BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()
        finally:
            image.close()
    finally:
        bitmap.close()


def _pdf_error(error: pdfium.PdfiumError) -> DocumentError:
    if error.err_code == 4:  # FPDF_ERR_PASSWORD
        return DocumentError("The PDF is password-protected. Upload an unlocked copy.")
    return DocumentError("The PDF could not be read. Export a new PDF or upload an image of the puzzle.")


def preview_pdf(data: bytes, name: str = "", content_type: str = "") -> dict:
    if upload_kind(data, name, content_type) != "pdf":
        raise DocumentError("Page previews require a PDF file.", 415)
    try:
        with _PDF_LOCK, pdfium.PdfDocument(data) as pdf:
            count = _page_count(pdf)
            pages = []
            for index in range(count):
                page = pdf[index]
                try:
                    png = _render(page, PREVIEW_SIDE)
                finally:
                    page.close()
                pages.append({"number": index + 1,
                              "preview": "data:image/png;base64," + base64.b64encode(png).decode("ascii")})
            return {"kind": "pdf", "name": name or "Puzzle.pdf", "page_count": count, "pages": pages}
    except pdfium.PdfiumError as error:
        raise _pdf_error(error) from error


def prepare_document(data: bytes, name: str = "", content_type: str = "",
                     pages: str | None = None) -> ReaderDocument:
    kind = upload_kind(data, name, content_type)
    if kind == "image":
        if pages is not None:
            raise DocumentError("Page selection is only available for PDF files.")
        return ReaderDocument(data)
    try:
        with _PDF_LOCK, pdfium.PdfDocument(data) as pdf:
            numbers = selected_pages(pages, _page_count(pdf))
            images, texts = [], []
            # Divide the text budget between selected pages so a large first
            # page cannot remove all clues from later pages.
            per_page = MAX_DOCUMENT_TEXT // len(numbers) - 32
            for number in numbers:
                page = pdf[number - 1]
                try:
                    images.append(_render(page, MAX_PAGE_SIDE))
                    textpage = page.get_textpage()
                    try:
                        count = min(textpage.count_chars(), per_page)
                        text = textpage.get_text_range(0, count)[:per_page].strip() if count else ""
                    finally:
                        textpage.close()
                    if text:
                        texts.append(f"[Page {number}]\n{text}")
                finally:
                    page.close()
            return ReaderDocument(images[0], images[1:], "\n\n".join(texts), kind="pdf")
    except pdfium.PdfiumError as error:
        raise _pdf_error(error) from error
