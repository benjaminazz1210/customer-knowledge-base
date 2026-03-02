import sys
from app.services.document_parser import DocumentParser
from app.services.text_chunker import TextChunker

def test_document_parser():
    print("Testing DocumentParser.parse")
    
    # Test .txt
    txt_content = b"Hello TXT"
    res1 = DocumentParser.parse(txt_content, "test.txt")
    assert res1 == "Hello TXT", f"Expected 'Hello TXT', got {res1}"
    
    # Test .md
    md_content = b"# Hello MD"
    res2 = DocumentParser.parse(md_content, "test.md")
    assert res2 == "# Hello MD", f"Expected '# Hello MD', got {res2}"
    
    # Test unsupported
    try:
        DocumentParser.parse(b"exe content", "test.exe")
        assert False, "Should have raised ValueError for .exe"
    except ValueError as e:
        assert "Unsupported file format: .exe" in str(e)
    
    print("DocumentParser tests passed!")

def test_text_chunker():
    print("Testing TextChunker.chunk")
    
    text = "123456789"
    chunks = TextChunker.chunk(text, chunk_size=5, overlap=2)
    expected = ["12345", "45678", "789"]
    assert chunks == expected, f"Expected {expected}, got {chunks}"
    
    # Edge case: empty string
    empty_chunks = TextChunker.chunk("", chunk_size=5, overlap=2)
    assert empty_chunks == [], f"Expected [], got {empty_chunks}"

    # Edge case: text smaller than chunk_size
    small_chunks = TextChunker.chunk("12", chunk_size=5, overlap=2)
    assert small_chunks == ["12"], f"Expected ['12'], got {small_chunks}"
    
    print("TextChunker tests passed!")

if __name__ == "__main__":
    test_document_parser()
    test_text_chunker()
