"""
文档分块策略 - Markdown 结构分块
"""
import os
import re
from typing import List, Dict


class MarkdownChunker:
    """基于 Markdown 结构的文档分块器"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_by_structure(self, filepath: str) -> List[Dict[str, str]]:
        """按 Markdown 结构分块"""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        filename = os.path.basename(filepath)
        first_line = content.split("\n")[0]
        url = first_line.replace("# ", "").strip() if first_line.startswith("# ") else filename

        if first_line.startswith("# http"):
            content = "\n".join(content.split("\n")[1:])

        chunks = self._split_by_headers(content, url, filename)

        print(f"✓ 文档分块完成: {filename} -> {len(chunks)} 个块")

        return chunks

    def _split_by_headers(self, content: str, url: str, filename: str) -> List[Dict[str, str]]:
        """按标题层级分割内容"""
        chunks = []

        header_pattern = r'^(#{1,6})\s+(.+)$'

        lines = content.split("\n")
        current_chunk_lines = []
        current_header = ""
        current_level = 0

        for line in lines:
            header_match = re.match(header_pattern, line)

            if header_match:
                level = len(header_match.group(1))
                header = header_match.group(2)

                if current_chunk_lines and (
                    level <= 2 or
                    len("\n".join(current_chunk_lines)) >= self.chunk_size
                ):
                    chunk_content = "\n".join(current_chunk_lines).strip()
                    if chunk_content:
                        chunks.append(self._create_chunk(
                            chunk_content,
                            current_header,
                            url,
                            filename
                        ))

                    current_chunk_lines = []
                    if chunks and self.chunk_overlap > 0:
                        prev_lines = chunks[-1]["content"].split("\n")
                        overlap_lines = []
                        for l in reversed(prev_lines):
                            if len("\n".join(overlap_lines)) + len(l) <= self.chunk_overlap:
                                overlap_lines.insert(0, l)
                            else:
                                break
                        if overlap_lines:
                            current_chunk_lines.extend(overlap_lines)

                current_header = header
                current_level = level

            current_chunk_lines.append(line)

        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines).strip()
            if chunk_content:
                chunks.append(self._create_chunk(
                    chunk_content,
                    current_header,
                    url,
                    filename
                ))

        if not chunks and content.strip():
            chunks.append(self._create_chunk(
                content.strip(),
                "内容",
                url,
                filename
            ))

        return chunks

    def _create_chunk(self, content: str, header: str, url: str, filename: str) -> Dict[str, str]:
        """创建文档块"""
        chunk_id = f"{filename}_{header}_{hash(content)}"

        return {
            "id": chunk_id,
            "content": content,
            "metadata": {
                "header": header,
                "url": url,
                "filename": filename,
                "chunk_size": len(content)
            }
        }

    def chunk_document(self, filepath: str) -> List[Dict[str, str]]:
        """分块单个文档"""
        return self.chunk_by_structure(filepath)

    def chunk_documents(self, filepaths: List[str]) -> List[Dict[str, str]]:
        """分块多个文档"""
        all_chunks = []
        for filepath in filepaths:
            chunks = self.chunk_document(filepath)
            all_chunks.extend(chunks)

        return all_chunks
