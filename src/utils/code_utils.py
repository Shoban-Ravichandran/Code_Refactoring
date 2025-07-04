"""
Utility functions for code processing and manipulation.
"""

import ast
import re
import tokenize
import io
from typing import List, Dict, Tuple, Optional, Set, Any
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class CodeProcessor:
    """Utility class for processing and analyzing code."""
    
    def __init__(self):
        self.keywords = {
            'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally',
            'def', 'class', 'return', 'yield', 'import', 'from', 'as', 'with'
        }
    
    def normalize_code(self, code: str) -> str:
        """Normalize code by removing comments and extra whitespace."""
        try:
            # Parse and unparse to normalize
            tree = ast.parse(code)
            return ast.unparse(tree)
        except:
            # Fallback to string processing
            lines = []
            for line in code.split('\n'):
                # Remove comments
                line = re.sub(r'#.*$', '', line)
                # Normalize whitespace
                line = re.sub(r'\s+', ' ', line.strip())
                if line:
                    lines.append(line)
            return '\n'.join(lines)
    
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """Extract function definitions from code."""
        try:
            tree = ast.parse(code)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'line_start': node.lineno,
                        'line_end': getattr(node, 'end_lineno', node.lineno),
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': [ast.unparse(d) for d in node.decorator_list],
                        'docstring': ast.get_docstring(node),
                        'is_async': isinstance(node, ast.AsyncFunctionDef),
                        'complexity': self._calculate_function_complexity(node)
                    }
                    functions.append(func_info)
            
            return functions
        except Exception as e:
            logger.warning(f"Error extracting functions: {e}")
            return []
    
    def extract_classes(self, code: str) -> List[Dict[str, Any]]:
        """Extract class definitions from code."""
        try:
            tree = ast.parse(code)
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                    class_info = {
                        'name': node.name,
                        'line_start': node.lineno,
                        'line_end': getattr(node, 'end_lineno', node.lineno),
                        'bases': [ast.unparse(base) for base in node.bases],
                        'decorators': [ast.unparse(d) for d in node.decorator_list],
                        'docstring': ast.get_docstring(node),
                        'methods': [m.name for m in methods],
                        'method_count': len(methods),
                        'attributes': self._extract_class_attributes(node)
                    }
                    classes.append(class_info)
            
            return classes
        except Exception as e:
            logger.warning(f"Error extracting classes: {e}")
            return []
    
    def extract_imports(self, code: str) -> Dict[str, List[str]]:
        """Extract import statements from code."""
        try:
            tree = ast.parse(code)
            imports = {'standard': [], 'from_imports': [], 'aliases': {}}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports['standard'].append(alias.name)
                        if alias.asname:
                            imports['aliases'][alias.asname] = alias.name
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports['from_imports'].append(f"{module}.{alias.name}")
                        if alias.asname:
                            imports['aliases'][alias.asname] = f"{module}.{alias.name}"
            
            return imports
        except Exception as e:
            logger.warning(f"Error extracting imports: {e}")
            return {'standard': [], 'from_imports': [], 'aliases': {}}
    
    def extract_variables(self, code: str) -> Dict[str, List[str]]:
        """Extract variable assignments from code."""
        try:
            tree = ast.parse(code)
            variables = {'assignments': [], 'global_vars': [], 'nonlocal_vars': []}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            variables['assignments'].append(target.id)
                        elif isinstance(target, ast.Attribute):
                            variables['assignments'].append(ast.unparse(target))
                
                elif isinstance(node, ast.Global):
                    variables['global_vars'].extend(node.names)
                
                elif isinstance(node, ast.Nonlocal):
                    variables['nonlocal_vars'].extend(node.names)
            
            return variables
        except Exception as e:
            logger.warning(f"Error extracting variables: {e}")
            return {'assignments': [], 'global_vars': [], 'nonlocal_vars': []}
    
    def get_code_metrics(self, code: str) -> Dict[str, int]:
        """Get basic code metrics."""
        lines = code.split('\n')
        
        metrics = {
            'total_lines': len(lines),
            'code_lines': len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            'comment_lines': len([line for line in lines if line.strip().startswith('#')]),
            'blank_lines': len([line for line in lines if not line.strip()]),
            'max_line_length': max(len(line) for line in lines) if lines else 0,
            'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0
        }
        
        try:
            tree = ast.parse(code)
            metrics.update({
                'functions': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                'classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                'imports': len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]),
                'complexity': self._calculate_total_complexity(tree)
            })
        except:
            metrics.update({'functions': 0, 'classes': 0, 'imports': 0, 'complexity': 1})
        
        return metrics
    
    def detect_code_smells(self, code: str) -> List[Dict[str, Any]]:
        """Detect common code smells."""
        smells = []
        
        try:
            tree = ast.parse(code)
            
            # Long methods
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    length = getattr(node, 'end_lineno', node.lineno) - node.lineno + 1
                    if length > 30:
                        smells.append({
                            'type': 'long_method',
                            'location': f"line {node.lineno}",
                            'description': f"Method '{node.name}' is {length} lines long",
                            'severity': 'medium' if length < 50 else 'high'
                        })
            
            # Large classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                    if len(methods) > 20:
                        smells.append({
                            'type': 'large_class',
                            'location': f"line {node.lineno}",
                            'description': f"Class '{node.name}' has {len(methods)} methods",
                            'severity': 'medium' if len(methods) < 30 else 'high'
                        })
            
            # Magic numbers
            smells.extend(self._detect_magic_numbers(tree))
            
            # Duplicate code
            smells.extend(self._detect_duplicate_code(code))
            
            # Complex conditionals
            smells.extend(self._detect_complex_conditionals(tree))
            
        except Exception as e:
            logger.warning(f"Error detecting code smells: {e}")
        
        return smells
    
    def format_code(self, code: str, style: str = "pep8") -> str:
        """Format code according to style guidelines."""
        # Basic formatting - in practice would use black or autopep8
        try:
            tree = ast.parse(code)
            formatted = ast.unparse(tree)
            
            # Apply basic formatting rules
            lines = []
            for line in formatted.split('\n'):
                # Fix indentation
                if line.strip():
                    indent_level = (len(line) - len(line.lstrip())) // 4
                    content = line.lstrip()
                    lines.append('    ' * indent_level + content)
                else:
                    lines.append('')
            
            return '\n'.join(lines)
        except:
            return code
    
    def extract_docstrings(self, code: str) -> Dict[str, str]:
        """Extract docstrings from functions and classes."""
        try:
            tree = ast.parse(code)
            docstrings = {}
            
            # Module docstring
            if (tree.body and isinstance(tree.body[0], ast.Expr) 
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
                docstrings['module'] = tree.body[0].value.value
            
            # Function and class docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        docstrings[f"{type(node).__name__}:{node.name}"] = docstring
            
            return docstrings
        except Exception as e:
            logger.warning(f"Error extracting docstrings: {e}")
            return {}
    
    def get_complexity_metrics(self, code: str) -> Dict[str, float]:
        """Calculate various complexity metrics."""
        try:
            tree = ast.parse(code)
            
            metrics = {
                'cyclomatic_complexity': self._calculate_total_complexity(tree),
                'halstead_complexity': self._calculate_halstead_complexity(code),
                'nesting_depth': self._calculate_max_nesting(tree),
                'cognitive_complexity': self._calculate_cognitive_complexity(tree)
            }
            
            return metrics
        except Exception as e:
            logger.warning(f"Error calculating complexity: {e}")
            return {'cyclomatic_complexity': 1, 'halstead_complexity': 1, 
                   'nesting_depth': 1, 'cognitive_complexity': 1}
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.Try, ast.ExceptHandler)):
                complexity += 1
        
        return complexity
    
    def _calculate_total_complexity(self, tree: ast.AST) -> int:
        """Calculate total cyclomatic complexity."""
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                complexity += 1
        
        return complexity
    
    def _calculate_halstead_complexity(self, code: str) -> float:
        """Calculate Halstead complexity metrics."""
        try:
            operators = set()
            operands = set()
            total_operators = 0
            total_operands = 0
            
            # Tokenize code
            tokens = tokenize.generate_tokens(io.StringIO(code).readline)
            
            for token in tokens:
                if token.type == tokenize.OP:
                    operators.add(token.string)
                    total_operators += 1
                elif token.type == tokenize.NAME and token.string not in self.keywords:
                    operands.add(token.string)
                    total_operands += 1
            
            # Calculate Halstead metrics
            n1 = len(operators)  # Unique operators
            n2 = len(operands)   # Unique operands
            N1 = total_operators # Total operators
            N2 = total_operands  # Total operands
            
            if n1 == 0 or n2 == 0:
                return 1.0
            
            vocabulary = n1 + n2
            length = N1 + N2
            difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 1
            
            return length * difficulty
        except:
            return 1.0
    
    def _calculate_max_nesting(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        def get_depth(node, current_depth=0):
            max_depth = current_depth
            
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            
            for child in ast.iter_child_nodes(node):
                max_depth = max(max_depth, get_depth(child, current_depth))
            
            return max_depth
        
        return get_depth(tree)
    
    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity."""
        complexity = 0
        nesting_level = 0
        
        def traverse(node, nesting=0):
            nonlocal complexity
            
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1 + nesting
                nesting += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                complexity += 1
            
            for child in ast.iter_child_nodes(node):
                traverse(child, nesting)
        
        traverse(tree)
        return complexity
    
    def _extract_class_attributes(self, node: ast.ClassDef) -> List[str]:
        """Extract class attributes."""
        attributes = []
        
        for child in node.body:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
            elif isinstance(child, ast.FunctionDef) and child.name == '__init__':
                # Extract self.attribute assignments
                for stmt in child.body:
                    if (isinstance(stmt, ast.Assign) and 
                        any(isinstance(t, ast.Attribute) and 
                            isinstance(t.value, ast.Name) and 
                            t.value.id == 'self' for t in stmt.targets)):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute):
                                attributes.append(target.attr)
        
        return attributes
    
    def _detect_magic_numbers(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect magic numbers in code."""
        magic_numbers = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                # Skip common non-magic numbers
                if node.value not in [0, 1, -1, 2, 10, 100]:
                    magic_numbers.append({
                        'type': 'magic_number',
                        'location': f"line {node.lineno}",
                        'description': f"Magic number {node.value} found",
                        'severity': 'low'
                    })
        
        return magic_numbers
    
    def _detect_duplicate_code(self, code: str) -> List[Dict[str, Any]]:
        """Detect potential duplicate code blocks."""
        duplicates = []
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        
        # Simple duplicate detection
        line_counts = defaultdict(list)
        for i, line in enumerate(lines):
            if len(line) > 10:  # Only consider substantial lines
                line_counts[line].append(i + 1)
        
        for line, line_numbers in line_counts.items():
            if len(line_numbers) > 1:
                duplicates.append({
                    'type': 'duplicate_code',
                    'location': f"lines {', '.join(map(str, line_numbers))}",
                    'description': f"Duplicate line found: {line[:50]}...",
                    'severity': 'medium' if len(line_numbers) > 2 else 'low'
                })
        
        return duplicates
    
    def _detect_complex_conditionals(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect overly complex conditional statements."""
        complex_conditionals = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Count nested conditions
                condition_complexity = self._count_boolean_operators(node.test)
                
                if condition_complexity > 3:
                    complex_conditionals.append({
                        'type': 'complex_conditional',
                        'location': f"line {node.lineno}",
                        'description': f"Complex conditional with {condition_complexity} boolean operators",
                        'severity': 'medium' if condition_complexity < 6 else 'high'
                    })
        
        return complex_conditionals
    
    def _count_boolean_operators(self, node: ast.AST) -> int:
        """Count boolean operators in an expression."""
        count = 0
        
        if isinstance(node, ast.BoolOp):
            count = len(node.values) - 1
            for value in node.values:
                count += self._count_boolean_operators(value)
        elif isinstance(node, ast.Compare):
            count = len(node.ops)
        else:
            for child in ast.iter_child_nodes(node):
                count += self._count_boolean_operators(child)
        
        return count

def similarity_score(code1: str, code2: str) -> float:
    """Calculate similarity score between two code snippets."""
    processor = CodeProcessor()
    
    # Normalize both codes
    norm1 = processor.normalize_code(code1)
    norm2 = processor.normalize_code(code2)
    
    if not norm1 or not norm2:
        return 0.0
    
    # Simple token-based similarity
    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())
    
    if not tokens1 and not tokens2:
        return 1.0
    
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    return len(intersection) / len(union) if union else 0.0

def extract_code_blocks(code: str, block_type: str = "function") -> List[str]:
    """Extract specific code blocks from source code."""
    try:
        tree = ast.parse(code)
        blocks = []
        
        for node in ast.walk(tree):
            if block_type == "function" and isinstance(node, ast.FunctionDef):
                block_code = ast.get_source_segment(code, node)
                if block_code:
                    blocks.append(block_code)
            elif block_type == "class" and isinstance(node, ast.ClassDef):
                block_code = ast.get_source_segment(code, node)
                if block_code:
                    blocks.append(block_code)
        
        return blocks
    except:
        return []

def validate_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Validate Python syntax."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def count_lines_of_code(code: str) -> Dict[str, int]:
    """Count different types of lines in code."""
    lines = code.split('\n')
    
    counts = {
        'total': len(lines),
        'code': 0,
        'comments': 0,
        'blank': 0,
        'docstring': 0
    }
    
    in_docstring = False
    docstring_delimiter = None
    
    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            counts['blank'] += 1
        elif stripped.startswith('#'):
            counts['comments'] += 1
        elif '"""' in stripped or "'''" in stripped:
            # Handle docstring detection
            if not in_docstring:
                in_docstring = True
                docstring_delimiter = '"""' if '"""' in stripped else "'''"
                counts['docstring'] += 1
            elif docstring_delimiter in stripped:
                in_docstring = False
                counts['docstring'] += 1
            else:
                counts['docstring'] += 1
        elif in_docstring:
            counts['docstring'] += 1
        else:
            counts['code'] += 1
    
    return counts

def find_unused_imports(code: str) -> List[str]:
    """Find unused import statements."""
    try:
        tree = ast.parse(code)
        
        # Collect all imports
        imports = set()
        import_nodes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imports.add(name)
                    import_nodes.append((node.lineno, alias.name, name))
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imports.add(name)
                    import_nodes.append((node.lineno, f"{node.module}.{alias.name}", name))
        
        # Collect all name usages
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Store):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # Get the base name
                current = node
                while isinstance(current, ast.Attribute):
                    current = current.value
                if isinstance(current, ast.Name):
                    used_names.add(current.id)
        
        # Find unused imports
        unused = []
        for line, full_name, used_name in import_nodes:
            if used_name not in used_names:
                unused.append(full_name)
        
        return unused
    except:
        return []

def extract_string_literals(code: str) -> List[str]:
    """Extract all string literals from code."""
    try:
        tree = ast.parse(code)
        strings = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                strings.append(node.value)
        
        return strings
    except:
        return []

def get_function_calls(code: str) -> List[Dict[str, Any]]:
    """Extract all function calls from code."""
    try:
        tree = ast.parse(code)
        calls = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = ast.unparse(node.func)
                
                calls.append({
                    'name': func_name,
                    'line': node.lineno,
                    'args': len(node.args),
                    'kwargs': len(node.keywords)
                })
        
        return calls
    except:
        return []

def remove_comments(code: str) -> str:
    """Remove comments from code while preserving structure."""
    lines = []
    for line in code.split('\n'):
        # Find comment start (but not in strings)
        in_string = False
        string_char = None
        comment_pos = None
        
        for i, char in enumerate(line):
            if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
            elif char == '#' and not in_string:
                comment_pos = i
                break
        
        if comment_pos is not None:
            line = line[:comment_pos].rstrip()
        
        lines.append(line)
    
    return '\n'.join(lines)

def get_variable_usage(code: str) -> Dict[str, Dict[str, int]]:
    """Analyze variable usage patterns."""
    try:
        tree = ast.parse(code)
        usage = defaultdict(lambda: {'reads': 0, 'writes': 0, 'locations': []})
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                var_name = node.id
                location = f"line {node.lineno}"
                
                if isinstance(node.ctx, ast.Store):
                    usage[var_name]['writes'] += 1
                elif isinstance(node.ctx, ast.Load):
                    usage[var_name]['reads'] += 1
                
                usage[var_name]['locations'].append(location)
        
        return dict(usage)
    except:
        return {}

def calculate_maintainability_index(code: str) -> float:
    """Calculate maintainability index using simplified formula."""
    try:
        processor = CodeProcessor()
        metrics = processor.get_code_metrics(code)
        complexity_metrics = processor.get_complexity_metrics(code)
        
        loc = metrics['code_lines']
        complexity = complexity_metrics['cyclomatic_complexity']
        halstead = complexity_metrics['halstead_complexity']
        
        if loc == 0:
            return 100.0
        
        # Simplified maintainability index
        mi = 171 - 5.2 * (halstead ** 0.23) - 0.23 * complexity - 16.2 * (loc ** 0.5)
        mi = max(0, min(100, mi))  # Clamp between 0 and 100
        
        return mi
    except:
        return 50.0  # Default middle value

def identify_refactoring_opportunities(code: str) -> List[Dict[str, Any]]:
    """Identify potential refactoring opportunities."""
    processor = CodeProcessor()
    opportunities = []
    
    try:
        # Get code smells
        smells = processor.detect_code_smells(code)
        
        # Convert smells to opportunities
        for smell in smells:
            if smell['type'] == 'long_method':
                opportunities.append({
                    'type': 'extract_method',
                    'location': smell['location'],
                    'description': 'Method is too long and could be broken down',
                    'priority': 'high' if smell['severity'] == 'high' else 'medium',
                    'estimated_effort': 'medium'
                })
            elif smell['type'] == 'magic_number':
                opportunities.append({
                    'type': 'replace_magic_numbers',
                    'location': smell['location'],
                    'description': 'Magic number should be replaced with named constant',
                    'priority': 'low',
                    'estimated_effort': 'low'
                })
            elif smell['type'] == 'duplicate_code':
                opportunities.append({
                    'type': 'extract_method',
                    'location': smell['location'],
                    'description': 'Duplicate code should be extracted to a method',
                    'priority': 'medium',
                    'estimated_effort': 'medium'
                })
            elif smell['type'] == 'complex_conditional':
                opportunities.append({
                    'type': 'simplify_conditional',
                    'location': smell['location'],
                    'description': 'Complex conditional should be simplified',
                    'priority': 'medium',
                    'estimated_effort': 'low'
                })
        
        # Check for other opportunities
        tree = ast.parse(code)
        
        # Check for classes with too many methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > 15:
                    opportunities.append({
                        'type': 'extract_class',
                        'location': f"line {node.lineno}",
                        'description': f"Class '{node.name}' has too many methods ({len(methods)})",
                        'priority': 'high',
                        'estimated_effort': 'high'
                    })
        
        # Check for unused imports
        unused_imports = find_unused_imports(code)
        if unused_imports:
            opportunities.append({
                'type': 'remove_dead_code',
                'location': 'imports',
                'description': f"Remove unused imports: {', '.join(unused_imports)}",
                'priority': 'low',
                'estimated_effort': 'low'
            })
        
    except Exception as e:
        logger.warning(f"Error identifying refactoring opportunities: {e}")
    
    return opportunities

def estimate_refactoring_effort(opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate total refactoring effort."""
    effort_weights = {
        'low': 1,
        'medium': 3,
        'high': 5
    }
    
    priority_weights = {
        'low': 1,
        'medium': 2,
        'high': 3
    }
    
    total_effort = 0
    total_priority = 0
    type_counts = defaultdict(int)
    
    for opp in opportunities:
        effort = effort_weights.get(opp.get('estimated_effort', 'medium'), 3)
        priority = priority_weights.get(opp.get('priority', 'medium'), 2)
        
        total_effort += effort
        total_priority += priority
        type_counts[opp['type']] += 1
    
    return {
        'total_opportunities': len(opportunities),
        'estimated_effort_points': total_effort,
        'average_priority': total_priority / len(opportunities) if opportunities else 0,
        'opportunity_types': dict(type_counts),
        'effort_distribution': {
            'low': sum(1 for o in opportunities if o.get('estimated_effort') == 'low'),
            'medium': sum(1 for o in opportunities if o.get('estimated_effort') == 'medium'),
            'high': sum(1 for o in opportunities if o.get('estimated_effort') == 'high')
        }
    }