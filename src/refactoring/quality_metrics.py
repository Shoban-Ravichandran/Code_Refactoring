"""
Quality metrics calculation for code assessment and improvement tracking.
"""

import ast
import re
import math
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MetricResult:
    """Represents a calculated metric result."""
    name: str
    value: float
    description: str
    range_info: Tuple[float, float]  # (min, max)
    interpretation: str
    category: str

class QualityMetrics:
    """Comprehensive quality metrics calculator."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.metrics_registry = self._initialize_metrics_registry()
    
    def calculate_all_metrics(self, code: str) -> Dict[str, MetricResult]:
        """Calculate all available quality metrics."""
        try:
            tree = ast.parse(code)
            results = {}
            
            # Complexity metrics
            results.update(self.calculate_complexity_metrics(code, tree))
            
            # Size metrics
            results.update(self.calculate_size_metrics(code, tree))
            
            # Maintainability metrics
            results.update(self.calculate_maintainability_metrics(code, tree))
            
            # Readability metrics
            results.update(self.calculate_readability_metrics(code, tree))
            
            # Design metrics
            results.update(self.calculate_design_metrics(tree))
            
            # Documentation metrics
            results.update(self.calculate_documentation_metrics(code, tree))
            
            # Testing metrics
            results.update(self.calculate_testing_metrics(tree))
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def calculate_complexity_metrics(self, code: str, tree: ast.AST) -> Dict[str, MetricResult]:
        """Calculate complexity-related metrics."""
        metrics = {}
        
        # Cyclomatic Complexity
        cc = self._calculate_cyclomatic_complexity(tree)
        metrics['cyclomatic_complexity'] = MetricResult(
            name="Cyclomatic Complexity",
            value=cc,
            description="Number of linearly independent paths through the code",
            range_info=(1, float('inf')),
            interpretation=self._interpret_cyclomatic_complexity(cc),
            category="complexity"
        )
        
        # Cognitive Complexity
        cognitive = self._calculate_cognitive_complexity(tree)
        metrics['cognitive_complexity'] = MetricResult(
            name="Cognitive Complexity",
            value=cognitive,
            description="Measure of how difficult code is to understand",
            range_info=(0, float('inf')),
            interpretation=self._interpret_cognitive_complexity(cognitive),
            category="complexity"
        )
        
        # Halstead Complexity
        halstead = self._calculate_halstead_metrics(code)
        metrics['halstead_difficulty'] = MetricResult(
            name="Halstead Difficulty",
            value=halstead['difficulty'],
            description="How difficult the program is to write or understand",
            range_info=(0, float('inf')),
            interpretation=self._interpret_halstead_difficulty(halstead['difficulty']),
            category="complexity"
        )
        
        metrics['halstead_effort'] = MetricResult(
            name="Halstead Effort",
            value=halstead['effort'],
            description="Mental effort required to develop the program",
            range_info=(0, float('inf')),
            interpretation=self._interpret_halstead_effort(halstead['effort']),
            category="complexity"
        )
        
        # Nesting Depth
        nesting = self._calculate_max_nesting_depth(tree)
        metrics['max_nesting_depth'] = MetricResult(
            name="Maximum Nesting Depth",
            value=nesting,
            description="Maximum depth of nested control structures",
            range_info=(0, float('inf')),
            interpretation=self._interpret_nesting_depth(nesting),
            category="complexity"
        )
        
        return metrics
    
    def calculate_size_metrics(self, code: str, tree: ast.AST) -> Dict[str, MetricResult]:
        """Calculate size-related metrics."""
        metrics = {}
        
        lines = code.split('\n')
        
        # Lines of Code metrics
        loc = len([line for line in lines if line.strip()])
        metrics['lines_of_code'] = MetricResult(
            name="Lines of Code",
            value=loc,
            description="Total number of non-empty lines",
            range_info=(0, float('inf')),
            interpretation=self._interpret_loc(loc),
            category="size"
        )
        
        # Source Lines of Code (excluding comments)
        sloc = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        metrics['source_lines_of_code'] = MetricResult(
            name="Source Lines of Code",
            value=sloc,
            description="Lines of code excluding comments and blank lines",
            range_info=(0, float('inf')),
            interpretation=self._interpret_sloc(sloc),
            category="size"
        )
        
        # Number of functions
        functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        metrics['number_of_functions'] = MetricResult(
            name="Number of Functions",
            value=functions,
            description="Total number of function definitions",
            range_info=(0, float('inf')),
            interpretation=self._interpret_function_count(functions),
            category="size"
        )
        
        # Number of classes
        classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        metrics['number_of_classes'] = MetricResult(
            name="Number of Classes",
            value=classes,
            description="Total number of class definitions",
            range_info=(0, float('inf')),
            interpretation=self._interpret_class_count(classes),
            category="size"
        )
        
        # Average function length
        if functions > 0:
            func_lengths = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    length = getattr(node, 'end_lineno', node.lineno) - node.lineno + 1
                    func_lengths.append(length)
            
            avg_func_length = sum(func_lengths) / len(func_lengths)
            metrics['average_function_length'] = MetricResult(
                name="Average Function Length",
                value=avg_func_length,
                description="Average number of lines per function",
                range_info=(1, float('inf')),
                interpretation=self._interpret_avg_function_length(avg_func_length),
                category="size"
            )
        
        return metrics
    
    def calculate_maintainability_metrics(self, code: str, tree: ast.AST) -> Dict[str, MetricResult]:
        """Calculate maintainability-related metrics."""
        metrics = {}
        
        # Maintainability Index
        mi = self._calculate_maintainability_index(code, tree)
        metrics['maintainability_index'] = MetricResult(
            name="Maintainability Index",
            value=mi,
            description="Composite metric indicating how maintainable the code is",
            range_info=(0, 100),
            interpretation=self._interpret_maintainability_index(mi),
            category="maintainability"
        )
        
        # Technical Debt Ratio
        td_ratio = self._calculate_technical_debt_ratio(code, tree)
        metrics['technical_debt_ratio'] = MetricResult(
            name="Technical Debt Ratio",
            value=td_ratio,
            description="Ratio of time to fix issues vs time to develop",
            range_info=(0, 1),
            interpretation=self._interpret_technical_debt_ratio(td_ratio),
            category="maintainability"
        )
        
        # Code Duplication
        duplication = self._calculate_code_duplication(code)
        metrics['code_duplication'] = MetricResult(
            name="Code Duplication",
            value=duplication,
            description="Percentage of duplicated code lines",
            range_info=(0, 1),
            interpretation=self._interpret_code_duplication(duplication),
            category="maintainability"
        )
        
        # Coupling
        coupling = self._calculate_coupling(tree)
        metrics['coupling'] = MetricResult(
            name="Coupling",
            value=coupling,
            description="Degree of interdependence between modules",
            range_info=(0, 1),
            interpretation=self._interpret_coupling(coupling),
            category="maintainability"
        )
        
        # Cohesion
        cohesion = self._calculate_cohesion(tree)
        metrics['cohesion'] = MetricResult(
            name="Cohesion",
            value=cohesion,
            description="Degree to which elements within a module work together",
            range_info=(0, 1),
            interpretation=self._interpret_cohesion(cohesion),
            category="maintainability"
        )
        
        return metrics
    
    def calculate_readability_metrics(self, code: str, tree: ast.AST) -> Dict[str, MetricResult]:
        """Calculate readability-related metrics."""
        metrics = {}
        
        # Comment Ratio
        lines = code.split('\n')
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        total_lines = len([line for line in lines if line.strip()])
        comment_ratio = comment_lines / max(total_lines, 1)
        
        metrics['comment_ratio'] = MetricResult(
            name="Comment Ratio",
            value=comment_ratio,
            description="Ratio of comment lines to total lines",
            range_info=(0, 1),
            interpretation=self._interpret_comment_ratio(comment_ratio),
            category="readability"
        )
        
        # Average Line Length
        non_empty_lines = [line for line in lines if line.strip()]
        avg_line_length = sum(len(line) for line in non_empty_lines) / max(len(non_empty_lines), 1)
        
        metrics['average_line_length'] = MetricResult(
            name="Average Line Length",
            value=avg_line_length,
            description="Average number of characters per line",
            range_info=(0, float('inf')),
            interpretation=self._interpret_avg_line_length(avg_line_length),
            category="readability"
        )
        
        # Naming Quality Score
        naming_score = self._calculate_naming_quality(tree)
        metrics['naming_quality'] = MetricResult(
            name="Naming Quality",
            value=naming_score,
            description="Quality of identifier naming conventions",
            range_info=(0, 1),
            interpretation=self._interpret_naming_quality(naming_score),
            category="readability"
        )
        
        # Indentation Consistency
        indentation_score = self._calculate_indentation_consistency(code)
        metrics['indentation_consistency'] = MetricResult(
            name="Indentation Consistency",
            value=indentation_score,
            description="Consistency of code indentation",
            range_info=(0, 1),
            interpretation=self._interpret_indentation_consistency(indentation_score),
            category="readability"
        )
        
        return metrics
    
    def calculate_design_metrics(self, tree: ast.AST) -> Dict[str, MetricResult]:
        """Calculate design-related metrics."""
        metrics = {}
        
        # Depth of Inheritance Tree
        dit = self._calculate_depth_of_inheritance(tree)
        metrics['depth_of_inheritance'] = MetricResult(
            name="Depth of Inheritance Tree",
            value=dit,
            description="Maximum inheritance depth in class hierarchy",
            range_info=(0, float('inf')),
            interpretation=self._interpret_inheritance_depth(dit),
            category="design"
        )
        
        # Number of Children
        noc = self._calculate_number_of_children(tree)
        metrics['number_of_children'] = MetricResult(
            name="Number of Children",
            value=noc,
            description="Average number of immediate subclasses",
            range_info=(0, float('inf')),
            interpretation=self._interpret_number_of_children(noc),
            category="design"
        )
        
        # Response for Class
        rfc = self._calculate_response_for_class(tree)
        metrics['response_for_class'] = MetricResult(
            name="Response for Class",
            value=rfc,
            description="Number of methods that can be invoked in response to a message",
            range_info=(0, float('inf')),
            interpretation=self._interpret_response_for_class(rfc),
            category="design"
        )
        
        # Lack of Cohesion in Methods
        lcom = self._calculate_lack_of_cohesion(tree)
        metrics['lack_of_cohesion'] = MetricResult(
            name="Lack of Cohesion in Methods",
            value=lcom,
            description="Measure of how poorly methods in a class are related",
            range_info=(0, 1),
            interpretation=self._interpret_lack_of_cohesion(lcom),
            category="design"
        )
        
        return metrics
    
    def calculate_documentation_metrics(self, code: str, tree: ast.AST) -> Dict[str, MetricResult]:
        """Calculate documentation-related metrics."""
        metrics = {}
        
        # Documentation Coverage
        doc_coverage = self._calculate_documentation_coverage(tree)
        metrics['documentation_coverage'] = MetricResult(
            name="Documentation Coverage",
            value=doc_coverage,
            description="Percentage of functions and classes with docstrings",
            range_info=(0, 1),
            interpretation=self._interpret_documentation_coverage(doc_coverage),
            category="documentation"
        )
        
        # Documentation Quality
        doc_quality = self._calculate_documentation_quality(tree)
        metrics['documentation_quality'] = MetricResult(
            name="Documentation Quality",
            value=doc_quality,
            description="Quality assessment of existing documentation",
            range_info=(0, 1),
            interpretation=self._interpret_documentation_quality(doc_quality),
            category="documentation"
        )
        
        return metrics
    
    def calculate_testing_metrics(self, tree: ast.AST) -> Dict[str, MetricResult]:
        """Calculate testing-related metrics."""
        metrics = {}
        
        # Testability Score
        testability = self._calculate_testability_score(tree)
        metrics['testability_score'] = MetricResult(
            name="Testability Score",
            value=testability,
            description="How easy the code is to test",
            range_info=(0, 1),
            interpretation=self._interpret_testability_score(testability),
            category="testing"
        )
        
        # Test Coverage Potential
        test_potential = self._calculate_test_coverage_potential(tree)
        metrics['test_coverage_potential'] = MetricResult(
            name="Test Coverage Potential",
            value=test_potential,
            description="Potential for achieving good test coverage",
            range_info=(0, 1),
            interpretation=self._interpret_test_coverage_potential(test_potential),
            category="testing"
        )
        
        return metrics
    
    def compare_metrics(self, metrics1: Dict[str, MetricResult], 
                       metrics2: Dict[str, MetricResult]) -> Dict[str, Dict[str, float]]:
        """Compare two sets of metrics."""
        comparison = {}
        
        for metric_name in set(metrics1.keys()) | set(metrics2.keys()):
            if metric_name in metrics1 and metric_name in metrics2:
                value1 = metrics1[metric_name].value
                value2 = metrics2[metric_name].value
                
                if value1 != 0:
                    change_percent = ((value2 - value1) / value1) * 100
                else:
                    change_percent = 100 if value2 > 0 else 0
                
                comparison[metric_name] = {
                    'before': value1,
                    'after': value2,
                    'change': value2 - value1,
                    'change_percent': change_percent,
                    'improvement': self._is_improvement(metric_name, value1, value2)
                }
        
        return comparison
    
    def get_metric_summary(self, metrics: Dict[str, MetricResult]) -> Dict[str, Any]:
        """Get a summary of metrics by category."""
        summary = defaultdict(list)
        
        for metric in metrics.values():
            summary[metric.category].append({
                'name': metric.name,
                'value': metric.value,
                'interpretation': metric.interpretation
            })
        
        return dict(summary)
    
    # Private calculation methods
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate McCabe's cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.With):
                complexity += 1
        
        return complexity
    
    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity."""
        complexity = 0
        
        def calculate_node_complexity(node, nesting_level=0):
            nonlocal complexity
            
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1 + nesting_level
                nesting_level += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.Break) or isinstance(node, ast.Continue):
                complexity += 1
            
            for child in ast.iter_child_nodes(node):
                calculate_node_complexity(child, nesting_level)
        
        calculate_node_complexity(tree)
        return complexity
    
    def _calculate_halstead_metrics(self, code: str) -> Dict[str, float]:
        """Calculate Halstead complexity metrics."""
        try:
            import tokenize
            import io
            
            operators = set()
            operands = set()
            total_operators = 0
            total_operands = 0
            
            # Keywords that count as operators
            operator_keywords = {'if', 'else', 'elif', 'for', 'while', 'try', 'except', 
                                'finally', 'def', 'class', 'return', 'yield', 'import', 
                                'from', 'as', 'with', 'and', 'or', 'not', 'in', 'is'}
            
            tokens = tokenize.generate_tokens(io.StringIO(code).readline)
            
            for token in tokens:
                if token.type == tokenize.OP:
                    operators.add(token.string)
                    total_operators += 1
                elif token.type == tokenize.NAME:
                    if token.string in operator_keywords:
                        operators.add(token.string)
                        total_operators += 1
                    else:
                        operands.add(token.string)
                        total_operands += 1
                elif token.type in (tokenize.NUMBER, tokenize.STRING):
                    operands.add(token.string)
                    total_operands += 1
            
            n1 = len(operators)  # Unique operators
            n2 = len(operands)   # Unique operands
            N1 = total_operators # Total operators
            N2 = total_operands  # Total operands
            
            if n1 == 0 or n2 == 0:
                return {'difficulty': 0, 'effort': 0, 'volume': 0}
            
            vocabulary = n1 + n2
            length = N1 + N2
            volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
            difficulty = (n1 / 2) * (N2 / n2)
            effort = difficulty * volume
            
            return {
                'difficulty': difficulty,
                'effort': effort,
                'volume': volume
            }
        
        except Exception:
            return {'difficulty': 0, 'effort': 0, 'volume': 0}
    
    def _calculate_max_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        def get_depth(node, current_depth=0):
            max_depth = current_depth
            
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try, 
                               ast.FunctionDef, ast.ClassDef, ast.AsyncFor, ast.AsyncWith)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            
            for child in ast.iter_child_nodes(node):
                max_depth = max(max_depth, get_depth(child, current_depth))
            
            return max_depth
        
        return get_depth(tree)
    
    def _calculate_maintainability_index(self, code: str, tree: ast.AST) -> float:
        """Calculate maintainability index."""
        try:
            lines = code.split('\n')
            loc = len([line for line in lines if line.strip()])
            
            complexity = self._calculate_cyclomatic_complexity(tree)
            halstead = self._calculate_halstead_metrics(code)
            
            if loc == 0:
                return 100.0
            
            # Simplified maintainability index formula
            mi = 171 - 5.2 * math.log(halstead['volume']) - 0.23 * complexity - 16.2 * math.log(loc)
            
            # Normalize to 0-100 scale
            mi = max(0, min(100, mi))
            
            return mi
        except:
            return 50.0
    
    def _calculate_technical_debt_ratio(self, code: str, tree: ast.AST) -> float:
        """Calculate technical debt ratio."""
        # Simplified calculation based on various factors
        complexity = self._calculate_cyclomatic_complexity(tree)
        duplication = self._calculate_code_duplication(code)
        
        # Estimate debt based on complexity and duplication
        debt_indicators = [
            min(complexity / 20, 1.0),  # Normalize complexity
            duplication,
            self._calculate_naming_violations(tree)
        ]
        
        return sum(debt_indicators) / len(debt_indicators)
    
    def _calculate_code_duplication(self, code: str) -> float:
        """Calculate code duplication ratio."""
        lines = [line.strip() for line in code.split('\n') 
                if line.strip() and not line.strip().startswith('#')]
        
        if len(lines) < 2:
            return 0.0
        
        line_counts = Counter(lines)
        duplicated_lines = sum(count - 1 for count in line_counts.values() if count > 1)
        
        return duplicated_lines / len(lines)
    
    def _calculate_coupling(self, tree: ast.AST) -> float:
        """Calculate coupling metric."""
        imports = len([node for node in ast.walk(tree) 
                      if isinstance(node, (ast.Import, ast.ImportFrom))])
        classes = len([node for node in ast.walk(tree) 
                      if isinstance(node, ast.ClassDef)])
        
        if classes == 0:
            return 0.0
        
        # Simple coupling metric based on imports per class
        coupling_ratio = imports / classes
        return min(1.0, coupling_ratio / 10.0)
    
    def _calculate_cohesion(self, tree: ast.AST) -> float:
        """Calculate cohesion metric."""
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        if not classes:
            return 1.0
        
        total_cohesion = 0.0
        
        for cls in classes:
            methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)]
            
            if not methods:
                continue
            
            # Simple cohesion calculation based on method interactions
            cohesion = self._calculate_class_cohesion(cls)
            total_cohesion += cohesion
        
        return total_cohesion / len(classes) if classes else 1.0
    
    def _calculate_class_cohesion(self, class_node: ast.ClassDef) -> float:
        """Calculate cohesion for a single class."""
        methods = [node for node in class_node.body if isinstance(node, ast.FunctionDef)]
        
        if len(methods) <= 1:
            return 1.0
        
        # Find instance variables
        instance_vars = set()
        for method in methods:
            if method.name == '__init__':
                for node in ast.walk(method):
                    if (isinstance(node, ast.Assign) and 
                        any(isinstance(target, ast.Attribute) and 
                            isinstance(target.value, ast.Name) and 
                            target.value.id == 'self' for target in node.targets)):
                        for target in node.targets:
                            if isinstance(target, ast.Attribute):
                                instance_vars.add(target.attr)
        
        if not instance_vars:
            return 0.5  # Default cohesion when no instance variables
        
        # Count method-variable relationships
        method_var_usage = 0
        for method in methods:
            used_vars = set()
            for node in ast.walk(method):
                if (isinstance(node, ast.Attribute) and 
                    isinstance(node.value, ast.Name) and 
                    node.value.id == 'self' and 
                    node.attr in instance_vars):
                    used_vars.add(node.attr)
            method_var_usage += len(used_vars)
        
        max_possible = len(methods) * len(instance_vars)
        return method_var_usage / max_possible if max_possible > 0 else 0.5
    
    def _calculate_naming_quality(self, tree: ast.AST) -> float:
        """Calculate naming quality score."""
        total_names = 0
        good_names = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_names += 1
                if self._is_good_function_name(node.name):
                    good_names += 1
            elif isinstance(node, ast.ClassDef):
                total_names += 1
                if self._is_good_class_name(node.name):
                    good_names += 1
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                total_names += 1
                if self._is_good_variable_name(node.id):
                    good_names += 1
        
        return good_names / max(total_names, 1)
    
    def _calculate_indentation_consistency(self, code: str) -> float:
        """Calculate indentation consistency score."""
        lines = code.split('\n')
        indentations = []
        
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indentations.append(indent)
        
        if not indentations:
            return 1.0
        
        # Check for consistent indentation (multiples of 4 or 2)
        consistent_4 = all(indent % 4 == 0 for indent in indentations)
        consistent_2 = all(indent % 2 == 0 for indent in indentations)
        
        if consistent_4:
            return 1.0
        elif consistent_2:
            return 0.8
        else:
            return 0.3
    
    def _calculate_naming_violations(self, tree: ast.AST) -> float:
        """Calculate naming convention violations."""
        total_names = 0
        violations = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_names += 1
                if not self._is_good_function_name(node.name):
                    violations += 1
            elif isinstance(node, ast.ClassDef):
                total_names += 1
                if not self._is_good_class_name(node.name):
                    violations += 1
        
        return violations / max(total_names, 1)
    
    def _calculate_depth_of_inheritance(self, tree: ast.AST) -> float:
        """Calculate depth of inheritance tree."""
        max_depth = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                depth = len(node.bases)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_number_of_children(self, tree: ast.AST) -> float:
        """Calculate average number of children per class."""
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        if not classes:
            return 0.0
        
        # This is a simplified calculation
        # In practice, would need to analyze inheritance relationships
        total_inheritance = sum(len(cls.bases) for cls in classes)
        
        return total_inheritance / len(classes)
    
    def _calculate_response_for_class(self, tree: ast.AST) -> float:
        """Calculate response for class metric."""
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        if not classes:
            return 0.0
        
        total_response = 0
        
        for cls in classes:
            methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)]
            calls = set()
            
            for method in methods:
                for node in ast.walk(method):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            calls.add(node.func.id)
                        elif isinstance(node.func, ast.Attribute):
                            calls.add(node.func.attr)
            
            total_response += len(methods) + len(calls)
        
        return total_response / len(classes)
    
    def _calculate_lack_of_cohesion(self, tree: ast.AST) -> float:
        """Calculate lack of cohesion in methods."""
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        if not classes:
            return 0.0
        
        total_lcom = 0.0
        
        for cls in classes:
            cohesion = self._calculate_class_cohesion(cls)
            lcom = 1.0 - cohesion
            total_lcom += lcom
        
        return total_lcom / len(classes)
    
    def _calculate_documentation_coverage(self, tree: ast.AST) -> float:
        """Calculate documentation coverage."""
        documentable_items = 0
        documented_items = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                documentable_items += 1
                if ast.get_docstring(node):
                    documented_items += 1
        
        return documented_items / max(documentable_items, 1)
    
    def _calculate_documentation_quality(self, tree: ast.AST) -> float:
        """Calculate documentation quality score."""
        total_docs = 0
        quality_score = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    total_docs += 1
                    quality_score += self._assess_docstring_quality(docstring)
        
        return quality_score / max(total_docs, 1)
    
    def _calculate_testability_score(self, tree: ast.AST) -> float:
        """Calculate testability score."""
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if not functions:
            return 1.0
        
        testable_functions = 0
        
        for func in functions:
            # Simple testability heuristics
            is_testable = True
            
            # Functions with too many parameters are harder to test
            if len(func.args.args) > 5:
                is_testable = False
            
            # Functions with high complexity are harder to test
            complexity = self._calculate_cyclomatic_complexity(func)
            if complexity > 10:
                is_testable = False
            
            # Functions that access global state are harder to test
            if self._uses_global_state(func):
                is_testable = False
            
            if is_testable:
                testable_functions += 1
        
        return testable_functions / len(functions)
    
    def _calculate_test_coverage_potential(self, tree: ast.AST) -> float:
        """Calculate test coverage potential."""
        # Simplified calculation based on code structure
        functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        
        # More modular code has higher test coverage potential
        modularity_score = (functions + classes * 2) / max(functions + classes, 1)
        
        return min(1.0, modularity_score / 2)
    
    # Helper methods for name validation
    def _is_good_function_name(self, name: str) -> bool:
        """Check if function name follows good practices."""
        return (name.islower() and 
                len(name) >= 3 and 
                not name.startswith('__') and
                ('_' in name or len(name) <= 15))
    
    def _is_good_class_name(self, name: str) -> bool:
        """Check if class name follows good practices."""
        return (name[0].isupper() and 
                len(name) >= 3 and 
                name.isalnum())
    
    def _is_good_variable_name(self, name: str) -> bool:
        """Check if variable name follows good practices."""
        common_short_names = {'i', 'j', 'k', 'x', 'y', 'z', 'n', 'm'}
        return ((len(name) > 1 or name in common_short_names) and 
                name.islower() and 
                name.isalnum() or '_' in name)
    
    def _uses_global_state(self, func: ast.FunctionDef) -> bool:
        """Check if function uses global state."""
        for node in ast.walk(func):
            if isinstance(node, ast.Global):
                return True
        return False
    
    def _assess_docstring_quality(self, docstring: str) -> float:
        """Assess quality of a docstring."""
        if not docstring:
            return 0.0
        
        score = 0.0
        
        # Length check
        if len(docstring) > 10:
            score += 0.3
        
        # Contains description
        if len(docstring.split('.')[0]) > 5:
            score += 0.3
        
        # Contains parameter info
        if 'Args:' in docstring or 'Parameters:' in docstring or ':param' in docstring:
            score += 0.2
        
        # Contains return info
        if 'Returns:' in docstring or 'Return:' in docstring or ':return' in docstring:
            score += 0.2
        
        return score
    
    def _is_improvement(self, metric_name: str, old_value: float, new_value: float) -> bool:
        """Determine if a metric change represents an improvement."""
        # Metrics where lower values are better
        lower_is_better = {
            'cyclomatic_complexity', 'cognitive_complexity', 'halstead_difficulty',
            'halstead_effort', 'max_nesting_depth', 'technical_debt_ratio',
            'code_duplication', 'coupling', 'lack_of_cohesion'
        }
        
        # Metrics where higher values are better
        higher_is_better = {
            'maintainability_index', 'cohesion', 'comment_ratio', 'naming_quality',
            'indentation_consistency', 'documentation_coverage', 'documentation_quality',
            'testability_score', 'test_coverage_potential'
        }
        
        if metric_name in lower_is_better:
            return new_value < old_value
        elif metric_name in higher_is_better:
            return new_value > old_value
        else:
            # For size metrics, depends on context
            return abs(new_value - old_value) < old_value * 0.1  # Within 10% is neutral
    
    def _initialize_metrics_registry(self) -> Dict[str, Dict[str, Any]]:
        """Initialize registry of all available metrics."""
        return {
            'complexity': {
                'cyclomatic_complexity': 'McCabe cyclomatic complexity',
                'cognitive_complexity': 'Cognitive complexity',
                'halstead_difficulty': 'Halstead difficulty',
                'halstead_effort': 'Halstead effort',
                'max_nesting_depth': 'Maximum nesting depth'
            },
            'size': {
                'lines_of_code': 'Lines of code',
                'source_lines_of_code': 'Source lines of code',
                'number_of_functions': 'Number of functions',
                'number_of_classes': 'Number of classes',
                'average_function_length': 'Average function length'
            },
            'maintainability': {
                'maintainability_index': 'Maintainability index',
                'technical_debt_ratio': 'Technical debt ratio',
                'code_duplication': 'Code duplication',
                'coupling': 'Coupling',
                'cohesion': 'Cohesion'
            },
            'readability': {
                'comment_ratio': 'Comment ratio',
                'average_line_length': 'Average line length',
                'naming_quality': 'Naming quality',
                'indentation_consistency': 'Indentation consistency'
            },
            'design': {
                'depth_of_inheritance': 'Depth of inheritance tree',
                'number_of_children': 'Number of children',
                'response_for_class': 'Response for class',
                'lack_of_cohesion': 'Lack of cohesion in methods'
            },
            'documentation': {
                'documentation_coverage': 'Documentation coverage',
                'documentation_quality': 'Documentation quality'
            },
            'testing': {
                'testability_score': 'Testability score',
                'test_coverage_potential': 'Test coverage potential'
            }
        }
    
    # Interpretation methods
    def _interpret_cyclomatic_complexity(self, value: float) -> str:
        """Interpret cyclomatic complexity value."""
        if value <= 10:
            return "Low complexity - easy to test and maintain"
        elif value <= 20:
            return "Moderate complexity - acceptable"
        elif value <= 50:
            return "High complexity - difficult to test"
        else:
            return "Very high complexity - consider refactoring"
    
    def _interpret_cognitive_complexity(self, value: float) -> str:
        """Interpret cognitive complexity value."""
        if value <= 5:
            return "Low cognitive load - easy to understand"
        elif value <= 15:
            return "Moderate cognitive load"
        elif value <= 25:
            return "High cognitive load - may be hard to understand"
        else:
            return "Very high cognitive load - consider simplifying"
    
    def _interpret_halstead_difficulty(self, value: float) -> str:
        """Interpret Halstead difficulty value."""
        if value <= 10:
            return "Easy to understand and modify"
        elif value <= 20:
            return "Moderate difficulty"
        elif value <= 30:
            return "Difficult to understand"
        else:
            return "Very difficult - prone to errors"
    
    def _interpret_halstead_effort(self, value: float) -> str:
        """Interpret Halstead effort value."""
        if value <= 1000:
            return "Low mental effort required"
        elif value <= 10000:
            return "Moderate mental effort"
        elif value <= 100000:
            return "High mental effort"
        else:
            return "Very high mental effort"
    
    def _interpret_nesting_depth(self, value: float) -> str:
        """Interpret nesting depth value."""
        if value <= 3:
            return "Good structure - easy to follow"
        elif value <= 5:
            return "Acceptable nesting level"
        elif value <= 7:
            return "Deep nesting - consider refactoring"
        else:
            return "Very deep nesting - hard to understand"
    
    def _interpret_loc(self, value: float) -> str:
        """Interpret lines of code value."""
        if value <= 100:
            return "Small module - easy to understand"
        elif value <= 500:
            return "Medium-sized module"
        elif value <= 1000:
            return "Large module - consider splitting"
        else:
            return "Very large module - difficult to maintain"
    
    def _interpret_sloc(self, value: float) -> str:
        """Interpret source lines of code value."""
        if value <= 80:
            return "Compact code - good for readability"
        elif value <= 400:
            return "Reasonable size"
        elif value <= 800:
            return "Large - consider modularization"
        else:
            return "Very large - needs refactoring"
    
    def _interpret_function_count(self, value: float) -> str:
        """Interpret function count value."""
        if value <= 10:
            return "Good number of functions"
        elif value <= 25:
            return "Reasonable function count"
        elif value <= 50:
            return "Many functions - ensure good organization"
        else:
            return "Very many functions - consider restructuring"
    
    def _interpret_class_count(self, value: float) -> str:
        """Interpret class count value."""
        if value == 0:
            return "Procedural code - no classes"
        elif value <= 5:
            return "Good number of classes"
        elif value <= 15:
            return "Reasonable class count"
        else:
            return "Many classes - ensure good design"
    
    def _interpret_avg_function_length(self, value: float) -> str:
        """Interpret average function length."""
        if value <= 10:
            return "Short functions - good for readability"
        elif value <= 25:
            return "Reasonable function length"
        elif value <= 50:
            return "Long functions - consider breaking down"
        else:
            return "Very long functions - needs refactoring"
    
    def _interpret_maintainability_index(self, value: float) -> str:
        """Interpret maintainability index value."""
        if value >= 85:
            return "Excellent maintainability"
        elif value >= 70:
            return "Good maintainability"
        elif value >= 50:
            return "Moderate maintainability"
        elif value >= 25:
            return "Poor maintainability"
        else:
            return "Very poor maintainability"
    
    def _interpret_technical_debt_ratio(self, value: float) -> str:
        """Interpret technical debt ratio."""
        if value <= 0.1:
            return "Low technical debt"
        elif value <= 0.2:
            return "Manageable technical debt"
        elif value <= 0.4:
            return "Significant technical debt"
        else:
            return "High technical debt - urgent attention needed"
    
    def _interpret_code_duplication(self, value: float) -> str:
        """Interpret code duplication value."""
        percentage = value * 100
        if percentage <= 5:
            return "Low duplication - acceptable"
        elif percentage <= 15:
            return "Moderate duplication"
        elif percentage <= 30:
            return "High duplication - consider refactoring"
        else:
            return "Very high duplication - immediate attention needed"
    
    def _interpret_coupling(self, value: float) -> str:
        """Interpret coupling value."""
        if value <= 0.2:
            return "Low coupling - good design"
        elif value <= 0.4:
            return "Moderate coupling"
        elif value <= 0.6:
            return "High coupling - consider decoupling"
        else:
            return "Very high coupling - poor design"
    
    def _interpret_cohesion(self, value: float) -> str:
        """Interpret cohesion value."""
        if value >= 0.8:
            return "High cohesion - excellent design"
        elif value >= 0.6:
            return "Good cohesion"
        elif value >= 0.4:
            return "Moderate cohesion"
        else:
            return "Low cohesion - consider restructuring"
    
    def _interpret_comment_ratio(self, value: float) -> str:
        """Interpret comment ratio value."""
        percentage = value * 100
        if percentage >= 20:
            return "Well commented code"
        elif percentage >= 10:
            return "Adequately commented"
        elif percentage >= 5:
            return "Minimal comments"
        else:
            return "Poorly commented - add more comments"
    
    def _interpret_avg_line_length(self, value: float) -> str:
        """Interpret average line length."""
        if value <= 80:
            return "Good line length - easy to read"
        elif value <= 100:
            return "Acceptable line length"
        elif value <= 120:
            return "Long lines - consider breaking"
        else:
            return "Very long lines - hard to read"
    
    def _interpret_naming_quality(self, value: float) -> str:
        """Interpret naming quality value."""
        percentage = value * 100
        if percentage >= 90:
            return "Excellent naming conventions"
        elif percentage >= 75:
            return "Good naming conventions"
        elif percentage >= 50:
            return "Acceptable naming"
        else:
            return "Poor naming - improve consistency"
    
    def _interpret_indentation_consistency(self, value: float) -> str:
        """Interpret indentation consistency."""
        if value >= 0.9:
            return "Consistent indentation"
        elif value >= 0.7:
            return "Mostly consistent indentation"
        else:
            return "Inconsistent indentation - fix formatting"
    
    def _interpret_inheritance_depth(self, value: float) -> str:
        """Interpret inheritance depth."""
        if value <= 2:
            return "Shallow inheritance - good design"
        elif value <= 4:
            return "Moderate inheritance depth"
        elif value <= 6:
            return "Deep inheritance - consider composition"
        else:
            return "Very deep inheritance - poor design"
    
    def _interpret_number_of_children(self, value: float) -> str:
        """Interpret number of children."""
        if value <= 3:
            return "Good inheritance structure"
        elif value <= 6:
            return "Moderate inheritance branching"
        else:
            return "Wide inheritance - consider restructuring"
    
    def _interpret_response_for_class(self, value: float) -> str:
        """Interpret response for class metric."""
        if value <= 20:
            return "Good class complexity"
        elif value <= 50:
            return "Moderate class complexity"
        else:
            return "Complex class - consider simplifying"
    
    def _interpret_lack_of_cohesion(self, value: float) -> str:
        """Interpret lack of cohesion value."""
        if value <= 0.2:
            return "High cohesion - good design"
        elif value <= 0.4:
            return "Moderate cohesion"
        elif value <= 0.6:
            return "Low cohesion - consider refactoring"
        else:
            return "Very low cohesion - poor design"
    
    def _interpret_documentation_coverage(self, value: float) -> str:
        """Interpret documentation coverage."""
        percentage = value * 100
        if percentage >= 90:
            return "Excellent documentation coverage"
        elif percentage >= 70:
            return "Good documentation coverage"
        elif percentage >= 50:
            return "Moderate documentation"
        else:
            return "Poor documentation - add more docstrings"
    
    def _interpret_documentation_quality(self, value: float) -> str:
        """Interpret documentation quality."""
        if value >= 0.8:
            return "High quality documentation"
        elif value >= 0.6:
            return "Good documentation quality"
        elif value >= 0.4:
            return "Moderate documentation quality"
        else:
            return "Poor documentation quality"
    
    def _interpret_testability_score(self, value: float) -> str:
        """Interpret testability score."""
        percentage = value * 100
        if percentage >= 80:
            return "Highly testable code"
        elif percentage >= 60:
            return "Good testability"
        elif percentage >= 40:
            return "Moderate testability"
        else:
            return "Poor testability - consider refactoring"
    
    def _interpret_test_coverage_potential(self, value: float) -> str:
        """Interpret test coverage potential."""
        percentage = value * 100
        if percentage >= 80:
            return "High potential for test coverage"
        elif percentage >= 60:
            return "Good test coverage potential"
        elif percentage >= 40:
            return "Moderate test coverage potential"
        else:
            return "Low test coverage potential"


def calculate_quality_score(metrics: Dict[str, MetricResult]) -> Dict[str, float]:
    """Calculate overall quality score from metrics."""
    category_scores = {}
    
    # Weight each category
    weights = {
        'complexity': 0.25,
        'maintainability': 0.25,
        'readability': 0.20,
        'design': 0.15,
        'documentation': 0.10,
        'testing': 0.05
    }
    
    for category, weight in weights.items():
        category_metrics = [m for m in metrics.values() if m.category == category]
        
        if category_metrics:
            # Normalize metrics to 0-1 scale
            normalized_values = []
            for metric in category_metrics:
                # Normalize based on interpretation
                normalized = _normalize_metric_value(metric)
                normalized_values.append(normalized)
            
            category_scores[category] = sum(normalized_values) / len(normalized_values)
        else:
            category_scores[category] = 0.5  # Default neutral score
    
    # Calculate overall score
    overall_score = sum(score * weights[category] for category, score in category_scores.items())
    
    category_scores['overall'] = overall_score
    return category_scores


def _normalize_metric_value(metric: MetricResult) -> float:
    """Normalize a metric value to 0-1 scale based on interpretation."""
    value = metric.value
    interpretation = metric.interpretation.lower()
    
    # Simple normalization based on interpretation keywords
    if any(word in interpretation for word in ['excellent', 'high', 'good']):
        return 0.8 + (0.2 * min(1.0, value / (metric.range_info[1] or 100)))
    elif any(word in interpretation for word in ['moderate', 'acceptable', 'reasonable']):
        return 0.6
    elif any(word in interpretation for word in ['poor', 'low', 'difficult']):
        return 0.3
    elif any(word in interpretation for word in ['very poor', 'very high', 'urgent']):
        return 0.1
    else:
        return 0.5  # Default neutral