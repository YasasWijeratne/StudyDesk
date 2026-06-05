import fitz
import os

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024

def validate_file(file_name: str, file_size: int) -> tuple[bool, str]:
    ext = os.path.splitext(file_name)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File type not allowed. Please upload a PDF or TXT file."

    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum size is 10MB."

    return True, "ok"


def parse_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    doc.close()
    return full_text.strip()


def parse_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8").strip()


def parse_file(file_name: str, file_bytes: bytes) -> tuple[bool, str]:
    is_valid, message = validate_file(file_name, len(file_bytes))
    if not is_valid:
        return False, message

    ext = os.path.splitext(file_name)[1].lower()

    if ext == ".pdf":
        text = parse_pdf(file_bytes)
    elif ext == ".txt":
        text = parse_txt(file_bytes)

    if not text:
        return False, "Could not extract any text from this file. It may be a scanned image PDF."

    return True, text