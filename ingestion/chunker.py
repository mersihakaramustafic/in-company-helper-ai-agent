from typing import List


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    if not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                # Hard-split oversized paragraphs with overlap
                for i in range(0, len(para), chunk_size - overlap):
                    piece = para[i : i + chunk_size].strip()
                    if piece:
                        chunks.append(piece)
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks
