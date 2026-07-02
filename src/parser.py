import os
import re
import logging
import fitz

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024
PDF_HEADER = b"%PDF"


def sanitize_filename(filename: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "", filename)


def validate_file(filename: str, file_bytes: bytes) -> tuple[bool, str]:
    ext = os.path.splitext(filename)[1].lower()

    if len(file_bytes) > MAX_FILE_SIZE:
        return False, "File exceeds the 10MB limit."

    if ext not in ALLOWED_EXTENSIONS:
        return False, "Only PDF files are allowed."

    if not file_bytes.startswith(PDF_HEADER):
        return False, "Only valid PDF files are allowed."

    return True, "ok"


def parse_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    try:
        pages = []

        for page in doc:
            text = page.get_text().strip()
            if text:
                pages.append(text)

        return "\n\n".join(pages).strip()

    finally:
        doc.close()


def parse_file(filename: str, file_bytes: bytes) -> tuple[bool, str]:
    safe_name = sanitize_filename(filename)

    is_valid, message = validate_file(safe_name, file_bytes)
    if not is_valid:
        return False, message

    try:
        text = parse_pdf(file_bytes)
    except Exception:
        logger.exception("PDF parsing failed")
        return False, "Unable to process this file."

    if not text:
        return (
            False,
            "No text could be extracted. The PDF may be image-only."
        )

    return True, text


def get_safe_filename(filename: str) -> str:
    return sanitize_filename(filename)