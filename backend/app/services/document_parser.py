import os
from pypdf import PdfReader
from docx import Document
from pptx import Presentation
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
        elif ext == '.pptx':
            return DocumentParser._parse_pptx(file_content)
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

    @staticmethod
    def _parse_pptx(content: bytes) -> str:
        prs = Presentation(io.BytesIO(content))
        slides_text = []
        for i, slide in enumerate(prs.slides):
            slide_parts = [f"--- 第 {i+1} 页 ---"]
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        line = para.text.strip()
                        if line:
                            slide_parts.append(line)
                # 提取表格内容
                if shape.has_table:
                    for row in shape.table.rows:
                        row_text = " | ".join(
                            cell.text.strip() for cell in row.cells if cell.text.strip()
                        )
                        if row_text:
                            slide_parts.append(row_text)
            slides_text.append("\n".join(slide_parts))
        return "\n\n".join(slides_text)
