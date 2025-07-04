"""
AST (Abstract Syntax Tree) utilities for code analysis and transformation.
"""

import ast
import copy
from typing import List, Dict, Any, Optional, Union, Set, Tuple
import logging

logger = logging.getLogger(__name__)

class ASTAnalyzer:
    """Utility class for AST analysis and manipulation."""
    
    def __init__(self):
        self.node_types = {
            'control_flow': (ast.If, ast.For, ast.While, ast.Try, ast.With),
            'definitions': (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef),
            'assignments': (ast.Assign, ast.AugAssign, ast.AnnAssign),
            'imports': (ast.Import, ast.ImportFrom),
            'expressions': (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call)
        }
    
    def parse_code(self, code: str) -> Optional[ast.AST]:
        """Safely parse code into AST."""
        try:
            return ast.parse(code)
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing code: {e}")
            return None
    
    def get_node_types(self, tree: ast.AST) -> Dict[str, int]:
        """Count different types of AST nodes."""
        counts = {}
        
        for node in ast.walk(tree):
            node_type = type(node).__name__
            counts[node_type] = counts.get(node_type, 0) + 1
        
        return counts
    
    def find_nodes_by_type(self, tree: ast.AST, node_type: type) -> List[ast.AST]:
        """Find all nodes of a specific type."""
        return [node for node in ast.walk(tree) if isinstance(node, node_type)]
    
    def get_function_info(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract detailed information about a function."""
        return {
            'name': func_node.name,
            'line_start': func_node.lineno,
            'line_end': getattr(func_node, 'end_lineno', func_node.lineno),
            'args': self._extract_function_args(func_node),
            'returns': self._extract_return_type(func_node),
            'decorators': [ast.unparse(d) for d in func_node.decorator_list],
            'docstring': ast.get_docstring(func_node),
            'is_async': isinstance(func_node, ast.AsyncFunctionDef),
            'complexity': self.calculate_complexity(func_node),
            'local_vars': self._extract_local_variables(func_node),
            'calls_made': self._extract_function_calls(func_node)
        }
    
    def get_class_info(self, class_node: ast.ClassDef) -> Dict[str, Any]:
        """Extract detailed information about a class."""
        methods = [node for node in class_node.body if isinstance(node, ast.FunctionDef)]
        
        return {
            'name': class_node.name,
            'line_start': class_node.lineno,
            'line_end': getattr(class_node, 'end_lineno', class_node.lineno),
            'bases': [ast.unparse(base) for base in class_node.bases],
            'decorators': [ast.unparse(d) for d in class_node.decorator_list],
            'docstring': ast.get_docstring(class_node),
            'methods': [self.get_function_info(method) for method in methods],
            'attributes': self._extract_class_attributes(class_node),
            'metaclass': self._extract_metaclass(class_node),
            'is_abstract': self._is_abstract_class(class_node)
        }
    
    def calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a node."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.With):
                complexity += 1
        
        return complexity
    
    def get_dependencies(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Extract import dependencies."""
        dependencies = {
            'standard_library': [],
            'third_party': [],
            'local': [],
            'from_imports': {}
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    dependencies['standard_library'].append(module_name)
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                imports = [alias.name for alias in node.names]
                dependencies['from_imports'][module] = imports
        
        return dependencies
    
    def find_unused_variables(self, tree: ast.AST) -> List[str]:
        """Find variables that are assigned but never used."""
        assigned = set()
        used = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    assigned.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    used.add(node.id)
        
        return list(assigned - used)
    
    def extract_string_patterns(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Extract string patterns for analysis."""
        patterns = {
            'string_literals': [],
            'format_strings': [],
            'raw_strings': [],
            'byte_strings': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                patterns['string_literals'].append(node.value)
            elif isinstance(node, ast.JoinedStr):  # f-strings
                patterns['format_strings'].append(ast.unparse(node))
        
        return patterns
    
    def get_control_flow_structure(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze control flow structure."""
        structure = {
            'if_statements': 0,
            'for_loops': 0,
            'while_loops': 0,
            'try_blocks': 0,
            'with_statements': 0,
            'max_nesting_depth': 0,
            'nested_structures': []
        }
        
        def analyze_nesting(node, depth=0):
            structure['max_nesting_depth'] = max(structure['max_nesting_depth'], depth)
            
            if isinstance(node, ast.If):
                structure['if_statements'] += 1
                if depth > 2:
                    structure['nested_structures'].append(f"Nested if at depth {depth}")
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                structure['for_loops'] += 1
                if depth > 2:
                    structure['nested_structures'].append(f"Nested for at depth {depth}")
            elif isinstance(node, ast.While):
                structure['while_loops'] += 1
            elif isinstance(node, ast.Try):
                structure['try_blocks'] += 1
            elif isinstance(node, ast.With):
                structure['with_statements'] += 1
            
            new_depth = depth + 1 if isinstance(node, self.node_types['control_flow']) else depth
            
            for child in ast.iter_child_nodes(node):
                analyze_nesting(child, new_depth)
        
        analyze_nesting(tree)
        return structure
    
    def extract_method_calls_graph(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Extract method call relationships."""
        call_graph = {}
        current_function = None
        
        class CallVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                nonlocal current_function
                old_function = current_function
                current_function = node.name
                call_graph[current_function] = []
                self.generic_visit(node)
                current_function = old_function
            
            def visit_Call(self, node):
                if current_function:
                    if isinstance(node.func, ast.Name):
                        call_graph[current_function].append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        call_graph[current_function].append(ast.unparse(node.func))
                self.generic_visit(node)
        
        visitor = CallVisitor()
        visitor.visit(tree)
        return call_graph
    
    def find_long_methods(self, tree: ast.AST, threshold: int = 30) -> List[Dict[str, Any]]:
        """Find methods longer than threshold."""
        long_methods = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                length = getattr(node, 'end_lineno', node.lineno) - node.lineno + 1
                if length > threshold:
                    long_methods.append({
                        'name': node.name,
                        'length': length,
                        'start_line': node.lineno,
                        'complexity': self.calculate_complexity(node)
                    })
        
        return long_methods
    
    def find_large_classes(self, tree: ast.AST, method_threshold: int = 20) -> List[Dict[str, Any]]:
        """Find classes with too many methods."""
        large_classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > method_threshold:
                    large_classes.append({
                        'name': node.name,
                        'method_count': len(methods),
                        'start_line': node.lineno,
                        'methods': [m.name for m in methods]
                    })
        
        return large_classes
    
    def compare_ast_structure(self, tree1: ast.AST, tree2: ast.AST) -> Dict[str, float]:
        """Compare structural similarity between two ASTs."""
        def get_structure_signature(tree):
            signature = []
            for node in ast.walk(tree):
                signature.append(type(node).__name__)
            return signature
        
        sig1 = get_structure_signature(tree1)
        sig2 = get_structure_signature(tree2)
        
        # Calculate similarity metrics
        common = len(set(sig1) & set(sig2))
        total = len(set(sig1) | set(sig2))
        
        structural_similarity = common / total if total > 0 else 0.0
        
        # Sequence similarity
        min_len = min(len(sig1), len(sig2))
        max_len = max(len(sig1), len(sig2))
        
        if min_len == 0:
            sequence_similarity = 0.0
        else:
            matches = sum(1 for i in range(min_len) if sig1[i] == sig2[i])
            sequence_similarity = matches / max_len
        
        return {
            'structural_similarity': structural_similarity,
            'sequence_similarity': sequence_similarity,
            'size_ratio': min_len / max_len if max_len > 0 else 0.0
        }
    
    def _extract_function_args(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract function argument information."""
        args_info = {
            'positional': [],
            'defaults': [],
            'varargs': None,
            'kwargs': None,
            'annotations': {}
        }
        
        args = func_node.args
        
        # Positional arguments
        for arg in args.args:
            args_info['positional'].append(arg.arg)
            if arg.annotation:
                args_info['annotations'][arg.arg] = ast.unparse(arg.annotation)
        
        # Default values
        if args.defaults:
            args_info['defaults'] = [ast.unparse(default) for default in args.defaults]
        
        # *args
        if args.vararg:
            args_info['varargs'] = args.vararg.arg
        
        # **kwargs
        if args.kwarg:
            args_info['kwargs'] = args.kwarg.arg
        
        return args_info
    
    def _extract_return_type(self, func_node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation."""
        if func_node.returns:
            return ast.unparse(func_node.returns)
        return None
    
    def _extract_local_variables(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract local variables from function."""
        local_vars = set()
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                local_vars.add(node.id)
        
        return list(local_vars)
    
    def _extract_function_calls(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract function calls made within a function."""
        calls = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(ast.unparse(node.func))
        
        return calls
    
    def _extract_class_attributes(self, class_node: ast.ClassDef) -> Dict[str, List[str]]:
        """Extract class attributes."""
        attributes = {
            'class_variables': [],
            'instance_variables': []
        }
        
        # Class variables (assignments at class level)
        for node in class_node.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        attributes['class_variables'].append(target.id)
        
        # Instance variables (self.attr in __init__)
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                for stmt in ast.walk(node):
                    if (isinstance(stmt, ast.Assign) and 
                        any(isinstance(t, ast.Attribute) and 
                            isinstance(t.value, ast.Name) and 
                            t.value.id == 'self' for t in stmt.targets)):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute):
                                attributes['instance_variables'].append(target.attr)
        
        return attributes
    
    def _extract_metaclass(self, class_node: ast.ClassDef) -> Optional[str]:
        """Extract metaclass information."""
        for keyword in class_node.keywords:
            if keyword.arg == 'metaclass':
                return ast.unparse(keyword.value)
        return None
    
    def _is_abstract_class(self, class_node: ast.ClassDef) -> bool:
        """Check if class is abstract."""
        # Check for ABC inheritance
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id in ['ABC', 'AbstractBase']:
                return True
        
        # Check for abstract methods
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                        return True
        
        return False

class ASTTransformer:
    """Utility class for AST transformations."""
    
    def __init__(self):
        self.analyzer = ASTAnalyzer()
    
    def extract_method(self, tree: ast.AST, method_name: str, lines_to_extract: Tuple[int, int]) -> Tuple[ast.AST, ast.AST]:
        """Extract method refactoring transformation."""
        start_line, end_line = lines_to_extract
        
        # Find the function containing the lines
        target_function = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.lineno <= start_line <= getattr(node, 'end_lineno', node.lineno):
                    target_function = node
                    break
        
        if not target_function:
            return tree, None
        
        # Create new method AST
        extracted_method = self._create_extracted_method(target_function, method_name, start_line, end_line)
        
        # Modify original method to call extracted method
        modified_tree = self._replace_with_method_call(tree, target_function, method_name, start_line, end_line)
        
        return modified_tree, extracted_method
    
    def inline_method(self, tree: ast.AST, method_name: str) -> ast.AST:
        """Inline method refactoring transformation."""
        # Find method to inline
        method_to_inline = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                method_to_inline = node
                break
        
        if not method_to_inline:
            return tree
        
        # Replace calls with method body
        transformer = MethodInliner(method_name, method_to_inline)
        return transformer.visit(tree)
    
    def extract_variable(self, tree: ast.AST, expression_node: ast.AST, variable_name: str) -> ast.AST:
        """Extract variable refactoring transformation."""
        transformer = VariableExtractor(expression_node, variable_name)
        return transformer.visit(tree)
    
    def rename_identifier(self, tree: ast.AST, old_name: str, new_name: str) -> ast.AST:
        """Rename identifier transformation."""
        transformer = IdentifierRenamer(old_name, new_name)
        return transformer.visit(tree)
    
    def _create_extracted_method(self, original_func: ast.FunctionDef, method_name: str, 
                                start_line: int, end_line: int) -> ast.FunctionDef:
        """Create a new method from extracted lines."""
        # Extract statements within line range
        extracted_stmts = []
        for stmt in original_func.body:
            if hasattr(stmt, 'lineno') and start_line <= stmt.lineno <= end_line:
                extracted_stmts.append(stmt)
        
        # Create new function
        new_func = ast.FunctionDef(
            name=method_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg='self', annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=extracted_stmts,
            decorator_list=[],
            returns=None,
            type_comment=None
        )
        
        return new_func
    
    def _replace_with_method_call(self, tree: ast.AST, target_func: ast.FunctionDef, 
                                 method_name: str, start_line: int, end_line: int) -> ast.AST:
        """Replace extracted lines with method call."""
        transformer = ExtractMethodTransformer(target_func.name, method_name, start_line, end_line)
        return transformer.visit(tree)

class MethodInliner(ast.NodeTransformer):
    """AST transformer to inline method calls."""
    
    def __init__(self, method_name: str, method_node: ast.FunctionDef):
        self.method_name = method_name
        self.method_node = method_node
    
    def visit_Call(self, node):
        # Check if this is a call to the method we want to inline
        if (isinstance(node.func, ast.Attribute) and 
            node.func.attr == self.method_name):
            
            # Replace with method body
            return self._inline_method_body(node)
        
        return self.generic_visit(node)
    
    def _inline_method_body(self, call_node: ast.Call) -> ast.AST:
        """Replace method call with its body."""
        # For simplicity, return the first statement of the method
        if self.method_node.body:
            return copy.deepcopy(self.method_node.body[0])
        return call_node

class VariableExtractor(ast.NodeTransformer):
    """AST transformer to extract variables."""
    
    def __init__(self, expression_node: ast.AST, variable_name: str):
        self.expression_node = expression_node
        self.variable_name = variable_name
        self.extracted = False
    
    def visit_Expr(self, node):
        # If this expression matches our target, extract it
        if ast.dump(node.value) == ast.dump(self.expression_node) and not self.extracted:
            # Create variable assignment
            assignment = ast.Assign(
                targets=[ast.Name(id=self.variable_name, ctx=ast.Store())],
                value=self.expression_node
            )
            
            # Create expression using the variable
            new_expr = ast.Expr(value=ast.Name(id=self.variable_name, ctx=ast.Load()))
            
            self.extracted = True
            return [assignment, new_expr]
        
        return self.generic_visit(node)

class IdentifierRenamer(ast.NodeTransformer):
    """AST transformer to rename identifiers."""
    
    def __init__(self, old_name: str, new_name: str):
        self.old_name = old_name
        self.new_name = new_name
    
    def visit_Name(self, node):
        if node.id == self.old_name:
            node.id = self.new_name
        return node
    
    def visit_FunctionDef(self, node):
        if node.name == self.old_name:
            node.name = self.new_name
        return self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        if node.name == self.old_name:
            node.name = self.new_name
        return self.generic_visit(node)

class ExtractMethodTransformer(ast.NodeTransformer):
    """AST transformer for extract method refactoring."""
    
    def __init__(self, func_name: str, new_method_name: str, start_line: int, end_line: int):
        self.func_name = func_name
        self.new_method_name = new_method_name
        self.start_line = start_line
        self.end_line = end_line
    
    def visit_FunctionDef(self, node):
        if node.name == self.func_name:
            # Replace statements in line range with method call
            new_body = []
            method_call_added = False
            
            for stmt in node.body:
                if hasattr(stmt, 'lineno') and self.start_line <= stmt.lineno <= self.end_line:
                    if not method_call_added:
                        # Add method call
                        method_call = ast.Expr(
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id='self', ctx=ast.Load()),
                                    attr=self.new_method_name,
                                    ctx=ast.Load()
                                ),
                                args=[],
                                keywords=[]
                            )
                        )
                        new_body.append(method_call)
                        method_call_added = True
                else:
                    new_body.append(stmt)
            
            node.body = new_body
        
        return self.generic_visit(node)

def ast_to_code(tree: ast.AST) -> str:
    """Convert AST back to source code."""
    try:
        return ast.unparse(tree)
    except Exception as e:
        logger.error(f"Error converting AST to code: {e}")
        return ""

def compare_asts(tree1: ast.AST, tree2: ast.AST) -> bool:
    """Compare two ASTs for structural equality."""
    return ast.dump(tree1) == ast.dump(tree2)

def find_ast_differences(tree1: ast.AST, tree2: ast.AST) -> List[Dict[str, Any]]:
    """Find differences between two ASTs."""
    differences = []
    
    dump1 = ast.dump(tree1)
    dump2 = ast.dump(tree2)
    
    if dump1 != dump2:
        # Simple difference detection
        nodes1 = [type(node).__name__ for node in ast.walk(tree1)]
        nodes2 = [type(node).__name__ for node in ast.walk(tree2)]
        
        # Count differences
        from collections import Counter
        count1 = Counter(nodes1)
        count2 = Counter(nodes2)
        
        for node_type in set(count1.keys()) | set(count2.keys()):
            diff = count2.get(node_type, 0) - count1.get(node_type, 0)
            if diff != 0:
                differences.append({
                    'node_type': node_type,
                    'difference': diff,
                    'description': f"{node_type} count changed by {diff}"
                })
    
    return differences

def extract_ast_features(tree: ast.AST) -> Dict[str, Any]:
    """Extract features from AST for analysis."""
    analyzer = ASTAnalyzer()
    
    features = {
        'node_counts': analyzer.get_node_types(tree),
        'complexity': analyzer.calculate_complexity(tree),
        'dependencies': analyzer.get_dependencies(tree),
        'control_flow': analyzer.get_control_flow_structure(tree),
        'functions': len(analyzer.find_nodes_by_type(tree, ast.FunctionDef)),
        'classes': len(analyzer.find_nodes_by_type(tree, ast.ClassDef)),
        'imports': len(analyzer.find_nodes_by_type(tree, ast.Import)) + 
                  len(analyzer.find_nodes_by_type(tree, ast.ImportFrom))
    }
    
    return features

def normalize_ast(tree: ast.AST) -> ast.AST:
    """Normalize AST by removing location information and formatting."""
    # Create a copy and remove location info
    normalized = copy.deepcopy(tree)
    
    for node in ast.walk(normalized):
        # Remove location attributes
        for attr in ['lineno', 'col_offset', 'end_lineno', 'end_col_offset']:
            if hasattr(node, attr):
                delattr(node, attr)
    
    return normalized

def merge_asts(tree1: ast.AST, tree2: ast.AST) -> ast.AST:
    """Merge two ASTs (simple concatenation of statements)."""
    if isinstance(tree1, ast.Module) and isinstance(tree2, ast.Module):
        merged = ast.Module(
            body=tree1.body + tree2.body,
            type_ignores=[]
        )
        return merged
    else:
        return tree1

def validate_ast_transformation(original: ast.AST, transformed: ast.AST) -> Dict[str, Any]:
    """Validate that AST transformation is valid."""
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'changes_summary': {}
    }
    
    try:
        # Check if transformed AST is syntactically valid
        ast_to_code(transformed)
        
        # Compare features
        orig_features = extract_ast_features(original)
        trans_features = extract_ast_features(transformed)
        
        # Check for major structural changes
        if abs(orig_features['functions'] - trans_features['functions']) > 2:
            validation_result['warnings'].append(
                f"Function count changed significantly: {orig_features['functions']} -> {trans_features['functions']}"
            )
        
        if abs(orig_features['classes'] - trans_features['classes']) > 1:
            validation_result['warnings'].append(
                f"Class count changed: {orig_features['classes']} -> {trans_features['classes']}"
            )
        
        validation_result['changes_summary'] = {
            'functions_delta': trans_features['functions'] - orig_features['functions'],
            'classes_delta': trans_features['classes'] - orig_features['classes'],
            'complexity_delta': trans_features['complexity'] - orig_features['complexity']
        }
        
    except Exception as e:
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"Transformation resulted in invalid AST: {e}")
    
    return validation_result

class ASTMatcher:
    """Utility for pattern matching in ASTs."""
    
    def __init__(self):
        self.patterns = {
            'long_method': self._match_long_method,
            'magic_number': self._match_magic_number,
            'duplicate_code': self._match_duplicate_code,
            'complex_conditional': self._match_complex_conditional
        }
    
    def find_pattern(self, tree: ast.AST, pattern_name: str) -> List[Dict[str, Any]]:
        """Find instances of a specific pattern in the AST."""
        if pattern_name not in self.patterns:
            return []
        
        return self.patterns[pattern_name](tree)
    
    def _match_long_method(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find long methods."""
        matches = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                length = getattr(node, 'end_lineno', node.lineno) - node.lineno + 1
                if length > 25:
                    matches.append({
                        'node': node,
                        'type': 'long_method',
                        'location': f"line {node.lineno}",
                        'details': {'length': length, 'name': node.name}
                    })
        
        return matches
    
    def _match_magic_number(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find magic numbers."""
        matches = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if node.value not in [0, 1, -1, 2, 10, 100]:
                    matches.append({
                        'node': node,
                        'type': 'magic_number',
                        'location': f"line {node.lineno}",
                        'details': {'value': node.value}
                    })
        
        return matches
    
    def _match_duplicate_code(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find potential duplicate code blocks."""
        # Simplified duplicate detection
        matches = []
        statements = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.stmt):
                stmt_str = ast.dump(node)
                statements.append((node, stmt_str))
        
        # Find duplicates
        seen = {}
        for node, stmt_str in statements:
            if stmt_str in seen:
                matches.append({
                    'node': node,
                    'type': 'duplicate_code',
                    'location': f"line {node.lineno}",
                    'details': {'duplicate_of': f"line {seen[stmt_str].lineno}"}
                })
            else:
                seen[stmt_str] = node
        
        return matches
    
    def _match_complex_conditional(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find complex conditional statements."""
        matches = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                complexity = self._count_boolean_complexity(node.test)
                if complexity > 3:
                    matches.append({
                        'node': node,
                        'type': 'complex_conditional',
                        'location': f"line {node.lineno}",
                        'details': {'complexity': complexity}
                    })
        
        return matches
    
    def _count_boolean_complexity(self, node: ast.AST) -> int:
        """Count boolean complexity in a condition."""
        if isinstance(node, ast.BoolOp):
            return len(node.values) - 1 + sum(self._count_boolean_complexity(v) for v in node.values)
        elif isinstance(node, ast.Compare):
            return len(node.ops)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return 1 + self._count_boolean_complexity(node.operand)
        else:
            return 0