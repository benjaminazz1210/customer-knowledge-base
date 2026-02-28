import os
from pypdf import PdfReader
from docx import Document
import io

class DocumentParser:
    @staticmethod
    def parse(file_content: bytes, filename: str) -> str:
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in ['.txt', '.md']:
            return file_content.decode('utf-8')
        elif ext == '.pdf':
            return DocumentParser._parse_pdf(file_content)
        elif ext == '.docx':
            return DocumentParser._parse_docx(file_content)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def _parse_pdf(content: bytes) -> str:
        reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    @staticmethod
    def _parse_docx(content: bytes) -> str:
        doc = Document(io.BytesIO(content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
