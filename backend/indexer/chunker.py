"""
AST-Aware Code Chunker using Tree-sitter.

Instead of splitting code by character count, we parse source files into
Abstract Syntax Trees and extract complete semantic units: functions, classes,
methods, interfaces. Each chunk is syntactically valid and self-contained.
"""

import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
from tree_sitter import Language, Parser
from dataclasses import dataclass, field
from typing import Optional
import hashlib


# ---------------------------------------------------------------------------
# Language Configuration
# ---------------------------------------------------------------------------

PY_LANGUAGE = Language(tspython.language())
JS_LANGUAGE = Language(tsjavascript.language())

LANGUAGE_MAP: dict[str, Language] = {
    '.py': PY_LANGUAGE,
    '.js': JS_LANGUAGE,
    '.jsx': JS_LANGUAGE,
    '.ts': JS_LANGUAGE,
    '.tsx': JS_LANGUAGE,
    '.html': JS_LANGUAGE,
}

CHUNK_NODE_TYPES: dict[str, list[str]] = {
    '.py': [
        'function_definition',
        'class_definition',
        'decorated_definition',
    ],
    '.js': [
        'function_declaration',
        'class_declaration',
        'arrow_function',
        'method_definition',
        'export_statement',
        'lexical_declaration',
        'variable_declaration',
    ],
    '.jsx': [
        'function_declaration',
        'class_declaration',
        'arrow_function',
        'method_definition',
        'export_statement',
        'lexical_declaration',
        'variable_declaration',
    ],
    '.ts': [
        'function_declaration',
        'class_declaration',
        'method_definition',
        'interface_declaration',
        'type_alias_declaration',
        'export_statement',
        'lexical_declaration',
        'variable_declaration',
    ],
    '.tsx': [
        'function_declaration',
        'class_declaration',
        'method_definition',
        'interface_declaration',
        'type_alias_declaration',
        'export_statement',
        'lexical_declaration',
        'variable_declaration',
    ],
    '.html': [],
}

SKIP_DIRS = {
    'node_modules', '.git', '__pycache__', 'venv', '.venv',
    'dist', 'build', '.next', 'out', '.eggs', 'egg-info',
    'vendor', 'third_party',
}

SKIP_FILES = {
    'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
}

MAX_CHUNK_CHARS = 8000
MIN_CHUNK_CHARS = 30


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    A single code chunk extracted from an AST.
    Represents a complete, syntactically valid code unit (function, class, method).
    """
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    node_type: str
    parent_class: Optional[str]
    language: str
    raw_text: str
    context_header: str
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.raw_text)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def make_chunk_id(file_path: str, start_line: int) -> str:
    """Generate a deterministic unique ID for a chunk (SHA256 of path + line)."""
    key = f"{file_path}:{start_line}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def extract_node_name(node, source_bytes: bytes) -> Optional[str]:
    """Extract the name identifier from a function/class AST node."""
    if node.type == 'decorated_definition':
        for child in node.children:
            if child.type in ('function_definition', 'class_definition'):
                return extract_node_name(child, source_bytes)
        return None

    if node.type == 'export_statement':
        for child in node.children:
            if child.type in ('function_declaration', 'class_declaration',
                              'interface_declaration', 'type_alias_declaration'):
                return extract_node_name(child, source_bytes)
        return None

    for child in node.children:
        if child.type in ('identifier', 'type_identifier', 'property_identifier'):
            return source_bytes[child.start_byte:child.end_byte].decode('utf-8')

    return None


def _is_class_like(node_type: str) -> bool:
    """Check if a node type represents a class/interface definition."""
    return node_type in ('class_definition', 'class_declaration', 'interface_declaration')


def _format_node_type(node_type: str) -> str:
    """Convert AST node type names to human-readable format."""
    return node_type.replace('_', ' ').title()


# ---------------------------------------------------------------------------
# Core Chunking Logic
# ---------------------------------------------------------------------------

def chunk_file(file_path: str, source_code: str, ext: str) -> list[Chunk]:
    """
    Parse a source file and return AST-aware chunks.

    Walks the AST and extracts complete semantic units (functions, classes).
    Each chunk is syntactically valid — we never split a function halfway.

    Args:
        file_path: Relative path to the file
        source_code: Full source code as a string
        ext: File extension (e.g., ".py", ".js")

    Returns:
        List of Chunk objects, one per meaningful code unit
    """
    if ext not in LANGUAGE_MAP:
        return []

    parser = Parser(LANGUAGE_MAP[ext])
    source_bytes = source_code.encode('utf-8')
    tree = parser.parse(source_bytes)

    target_types = CHUNK_NODE_TYPES.get(ext, [])
    chunks: list[Chunk] = []

    def walk(node, parent_class_name: Optional[str] = None):
        if node.type in target_types:
            raw_text = source_bytes[node.start_byte:node.end_byte].decode('utf-8')

            if len(raw_text) < MIN_CHUNK_CHARS:
                return

            node_name = extract_node_name(node, source_bytes)
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            current_class = None
            if _is_class_like(node.type):
                current_class = node_name

            if parent_class_name:
                context_header = (
                    f"File: {file_path} | "
                    f"Class: {parent_class_name} | "
                    f"Method: {node_name} | "
                    f"Lines: {start_line}-{end_line}"
                )
            else:
                context_header = (
                    f"File: {file_path} | "
                    f"{_format_node_type(node.type)}: {node_name} | "
                    f"Lines: {start_line}-{end_line}"
                )

            chunk = Chunk(
                chunk_id=make_chunk_id(file_path, start_line),
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                node_type=node.type,
                parent_class=parent_class_name,
                language=ext.lstrip('.'),
                raw_text=raw_text,
                context_header=context_header,
            )
            chunks.append(chunk)

            for child in node.children:
                walk(child, parent_class_name=current_class or parent_class_name)
        else:
            for child in node.children:
                walk(child, parent_class_name=parent_class_name)

    walk(tree.root_node)
    return chunks


def chunk_file_with_fallback(
    file_path: str,
    source_code: str,
    ext: str,
    fallback_chunk_size: int = 1500
) -> list[Chunk]:
    """
    Try AST chunking first. Falls back to line-based chunking if no AST chunks
    are produced (e.g., config files, markdown, unsupported languages).
    """
    chunks = chunk_file(file_path, source_code, ext)

    if chunks:
        return chunks

    lines = source_code.split('\n')
    fallback_chunks = []
    current_chunk_lines = []
    current_char_count = 0
    chunk_start_line = 1

    for i, line in enumerate(lines, start=1):
        current_chunk_lines.append(line)
        current_char_count += len(line) + 1

        if current_char_count >= fallback_chunk_size:
            raw_text = '\n'.join(current_chunk_lines)
            if len(raw_text.strip()) >= MIN_CHUNK_CHARS:
                fallback_chunks.append(Chunk(
                    chunk_id=make_chunk_id(file_path, chunk_start_line),
                    file_path=file_path,
                    start_line=chunk_start_line,
                    end_line=i,
                    node_type='text_block',
                    parent_class=None,
                    language=ext.lstrip('.'),
                    raw_text=raw_text,
                    context_header=f"File: {file_path} | Lines: {chunk_start_line}-{i}",
                ))
            current_chunk_lines = []
            current_char_count = 0
            chunk_start_line = i + 1

    if current_chunk_lines:
        raw_text = '\n'.join(current_chunk_lines)
        if len(raw_text.strip()) >= MIN_CHUNK_CHARS:
            end_line = len(lines)
            fallback_chunks.append(Chunk(
                chunk_id=make_chunk_id(file_path, chunk_start_line),
                file_path=file_path,
                start_line=chunk_start_line,
                end_line=end_line,
                node_type='text_block',
                parent_class=None,
                language=ext.lstrip('.'),
                raw_text=raw_text,
                context_header=f"File: {file_path} | Lines: {chunk_start_line}-{end_line}",
            ))

    return fallback_chunks


def should_skip_path(path_str: str) -> bool:
    """Check if a file path should be skipped during indexing."""
    from pathlib import PurePosixPath
    parts = PurePosixPath(path_str).parts

    for part in parts:
        if part in SKIP_DIRS:
            return True

    if parts and parts[-1] in SKIP_FILES:
        return True

    return False
