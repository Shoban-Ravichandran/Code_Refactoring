"""
Utility functions and classes for code analysis and processing.
"""

from .code_utils import CodeProcessor, similarity_score, extract_code_blocks
from .ast_utils import ASTAnalyzer, ASTTransformer, ast_to_code, compare_asts
from .evaluation import RefactoringEvaluator

__all__ = [
    'CodeProcessor',
    'similarity_score',
    'extract_code_blocks',
    'ASTAnalyzer',
    'ASTTransformer',
    'ast_to_code',
    'compare_asts',
    'RefactoringEvaluator'
]