"""
Две стратегии разбиения документов на чанки:
  1. FixedSizeChunker — фиксированный размер в токенах с перекрытием
  2. StructuralChunker — по структуре документа (Markdown-заголовки, Python AST)
"""

import re
import ast
import tiktoken
from abc import ABC, abstractmethod


class Chunk:
    """Один чанк текста с метаданными."""

    __slots__ = ("text", "metadata")

    def __init__(self, text, metadata=None):
        self.text = text
        self.metadata = metadata or {}

    def __repr__(self):
        tok = self.metadata.get("token_count", "?")
        return f"Chunk(tokens={tok}, meta={list(self.metadata.keys())})"


class BaseChunker(ABC):
    """Базовый интерфейс чанкера."""

    @abstractmethod
    def chunk(self, text, source_file="", title=""):
        """Разбить текст на список Chunk."""

    @property
    @abstractmethod
    def strategy_name(self):
        ...


_encoder = None

def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def _token_len(text):
    return len(_get_encoder().encode(text))


class FixedSizeChunker(BaseChunker):
    """Разбивает текст на чанки фиксированного размера в токенах."""

    def __init__(self, chunk_size=256, overlap_tokens=64):
        self.chunk_size = chunk_size
        self.overlap_tokens = overlap_tokens

    @property
    def strategy_name(self):
        return "fixed_size"

    def chunk(self, text, source_file="", title=""):
        enc = _get_encoder()
        tokens = enc.encode(text)
        chunks = []
        start = 0
        idx = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)

            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    "source_file": source_file,
                    "title": title,
                    "strategy": self.strategy_name,
                    "chunk_index": idx,
                    "start_offset": start,
                    "token_count": len(chunk_tokens),
                },
            ))
            idx += 1
            start += self.chunk_size - self.overlap_tokens
            if start >= end:
                break

        return chunks


class StructuralChunker(BaseChunker):
    """Разбивает текст по структуре документа."""

    def __init__(self, max_chunk_tokens=512):
        self.max_chunk_tokens = max_chunk_tokens
        self._fallback = FixedSizeChunker(chunk_size=max_chunk_tokens, overlap_tokens=64)

    @property
    def strategy_name(self):
        return "structural"

    def chunk(self, text, source_file="", title=""):
        ext = source_file.rsplit(".", 1)[-1].lower() if "." in source_file else ""

        if ext in ("md", "markdown"):
            sections = self._split_markdown(text)
        elif ext == "py":
            sections = self._split_python(text)
        else:
            sections = self._split_paragraphs(text)

        chunks = []
        for idx, (section_title, section_text) in enumerate(sections):
            if not section_text.strip():
                continue

            tok_count = _token_len(section_text)

            if tok_count > self.max_chunk_tokens:
                sub_chunks = self._fallback.chunk(section_text, source_file, title)
                for sc in sub_chunks:
                    sc.metadata["strategy"] = self.strategy_name
                    sc.metadata["section_title"] = section_title
                    sc.metadata["chunk_index"] = len(chunks)
                    chunks.append(sc)
            else:
                chunks.append(Chunk(
                    text=section_text,
                    metadata={
                        "source_file": source_file,
                        "title": title or section_title,
                        "strategy": self.strategy_name,
                        "section_title": section_title,
                        "chunk_index": idx,
                        "start_offset": 0,
                        "token_count": tok_count,
                    },
                ))

        return chunks

    @staticmethod
    def _split_markdown(text):
        header_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        sections = []
        last_pos = 0
        last_title = "(intro)"

        for m in header_re.finditer(text):
            if m.start() > last_pos:
                sections.append((last_title, text[last_pos:m.start()]))
            last_title = m.group(2).strip()
            last_pos = m.start()

        if last_pos < len(text):
            sections.append((last_title, text[last_pos:]))

        return sections if sections else [("(document)", text)]

    @staticmethod
    def _split_python(text):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return [("(unparseable)", text)]

        lines = text.splitlines(keepends=True)
        sections = []
        last_end = 0

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = node.lineno - 1
                end_line = node.end_lineno

                if start_line > last_end:
                    preamble = "".join(lines[last_end:start_line])
                    if preamble.strip():
                        sections.append(("(module-level)", preamble))

                body = "".join(lines[start_line:end_line])
                name = f"{type(node).__name__}: {node.name}"
                sections.append((name, body))
                last_end = end_line

        if last_end < len(lines):
            tail = "".join(lines[last_end:])
            if tail.strip():
                sections.append(("(module-tail)", tail))

        return sections if sections else [("(module)", text)]

    @staticmethod
    def _split_paragraphs(text):
        parts = re.split(r"\n{2,}", text)
        return [(f"(paragraph-{i})", p) for i, p in enumerate(parts) if p.strip()]
