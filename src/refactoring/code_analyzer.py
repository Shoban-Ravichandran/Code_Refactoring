"""
Code analyzer for identifying refactoring opportunities and code quality issues.
"""

import ast
import re
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from ..utils.ast_utils import ASTAnalyzer, ASTMatcher
from ..utils.code_utils import CodeProcessor

logger = logging.getLogger(__name__)

@dataclass
class RefactoringOpportunity:
    """Represents a refactoring opportunity."""
    type: str
    location: str
    description: str
    priority: str  # low, medium, high
    effort: str    # low, medium, high
    confidence: float
    details: Dict[str, Any]
    code_snippet: str = ""

@dataclass
class QualityIssue:
    """Represents a code quality issue."""
    type: str
    severity: str  # low, medium, high, critical
    location: str
    description: str
    suggestion: str
    details: Dict[str, Any]

class CodeAnalyzer:
    """Analyzes code for refactoring opportunities and quality issues."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.ast_analyzer = ASTAnalyzer()
        self.ast_matcher = ASTMatcher()
        self.code_processor = CodeProcessor()
        
        # Thresholds for analysis
        self.thresholds = {
            'method_length': self.config.get('method_length_threshold', 25),
            'class_methods': self.config.get('class_methods_threshold', 15),
            'cyclomatic_complexity': self.config.get('complexity_threshold', 10),
            'nesting_depth': self.config.get('nesting_threshold', 4),
            'parameter_count': self.config.get('parameter_threshold', 5),
            'duplicate_lines': self.config.get('duplicate_threshold', 3)
        }
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Perform comprehensive code analysis."""
        logger.info("Starting comprehensive code analysis")
        
        try:
            tree = self.ast_analyzer.parse_code(code)
            if not tree:
                return {"error": "Could not parse code"}
            
            analysis_result = {
                "refactoring_opportunities": self.find_refactoring_opportunities(code, tree),
                "quality_issues": self.find_quality_issues(code, tree),
                "code_metrics": self.calculate_code_metrics(code, tree),
                "complexity_analysis": self.analyze_complexity(tree),
                "design_analysis": self.analyze_design_quality(tree),
                "maintainability_score": self.calculate_maintainability_score(code, tree),
                "recommendations": self.generate_recommendations(code, tree)
            }
            
            logger.info("Code analysis completed successfully")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error during code analysis: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def find_refactoring_opportunities(self, code: str, tree: ast.AST) -> List[RefactoringOpportunity]:
        """Find refactoring opportunities in the code."""
        opportunities = []
        
        # Extract Method opportunities
        opportunities.extend(self._find_extract_method_opportunities(tree))
        
        # Extract Variable opportunities
        opportunities.extend(self._find_extract_variable_opportunities(tree))
        
        # Replace Magic Numbers opportunities
        opportunities.extend(self._find_magic_number_opportunities(tree))
        
        # Simplify Conditional opportunities
        opportunities.extend(self._find_conditional_simplification_opportunities(tree))
        
        # Remove Dead Code opportunities
        opportunities.extend(self._find_dead_code_opportunities(code, tree))
        
        # Extract Class opportunities
        opportunities.extend(self._find_extract_class_opportunities(tree))
        
        # Inline Method opportunities
        opportunities.extend(self._find_inline_method_opportunities(tree))
        
        # Move Method opportunities
        opportunities.extend(self._find_move_method_opportunities(tree))
        
        # Sort by priority and confidence
        opportunities.sort(key=lambda x: (
            {'high': 3, 'medium': 2, 'low': 1}[x.priority],
            x.confidence
        ), reverse=True)
        
        return opportunities
    
    def find_quality_issues(self, code: str, tree: ast.AST) -> List[QualityIssue]:
        """Find code quality issues."""
        issues = []
        
        # Code smells
        smells = self.code_processor.detect_code_smells(code)
        for smell in smells:
            issues.append(QualityIssue(
                type=smell['type'],
                severity=smell['severity'],
                location=smell['location'],
                description=smell['description'],
                suggestion=self._get_smell_suggestion(smell['type']),
                details=smell
            ))
        
        # Naming convention issues
        issues.extend(self._find_naming_issues(tree))
        
        # Documentation issues
        issues.extend(self._find_documentation_issues(tree))
        
        # Performance issues
        issues.extend(self._find_performance_issues(tree))
        
        # Security issues
        issues.extend(self._find_security_issues(tree))
        
        return issues
    
    def calculate_code_metrics(self, code: str, tree: ast.AST) -> Dict[str, Any]:
        """Calculate comprehensive code metrics."""
        basic_metrics = self.code_processor.get_code_metrics(code)
        complexity_metrics = self.code_processor.get_complexity_metrics(code)
        
        # Additional metrics
        functions = self.ast_analyzer.find_nodes_by_type(tree, ast.FunctionDef)
        classes = self.ast_analyzer.find_nodes_by_type(tree, ast.ClassDef)
        
        function_metrics = []
        for func in functions:
            func_info = self.ast_analyzer.get_function_info(func)
            function_metrics.append({
                'name': func_info['name'],
                'complexity': func_info['complexity'],
                'length': func_info['line_end'] - func_info['line_start'] + 1,
                'parameter_count': len(func_info['args']['positional'])
            })
        
        class_metrics = []
        for cls in classes:
            cls_info = self.ast_analyzer.get_class_info(cls)
            class_metrics.append({
                'name': cls_info['name'],
                'method_count': len(cls_info['methods']),
                'attribute_count': len(cls_info['attributes']['class_variables']) + 
                                len(cls_info['attributes']['instance_variables']),
                'inheritance_depth': len(cls_info['bases'])
            })
        
        return {
            'basic_metrics': basic_metrics,
            'complexity_metrics': complexity_metrics,
            'function_metrics': function_metrics,
            'class_metrics': class_metrics,
            'duplication_score': self._calculate_duplication_score(code),
            'coupling_score': self._calculate_coupling_score(tree),
            'cohesion_score': self._calculate_cohesion_score(tree)
        }
    
    def analyze_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code complexity in detail."""
        complexity_analysis = {
            'overall_complexity': self.ast_analyzer.calculate_complexity(tree),
            'function_complexities': [],
            'complex_functions': [],
            'nesting_analysis': {},
            'control_flow_analysis': {}
        }
        
        # Analyze function complexities
        functions = self.ast_analyzer.find_nodes_by_type(tree, ast.FunctionDef)
        for func in functions:
            complexity = self.ast_analyzer.calculate_complexity(func)
            func_analysis = {
                'name': func.name,
                'complexity': complexity,
                'is_complex': complexity > self.thresholds['cyclomatic_complexity']
            }
            complexity_analysis['function_complexities'].append(func_analysis)
            
            if complexity > self.thresholds['cyclomatic_complexity']:
                complexity_analysis['complex_functions'].append(func_analysis)
        
        # Nesting analysis
        complexity_analysis['nesting_analysis'] = self.ast_analyzer.get_control_flow_structure(tree)
        
        return complexity_analysis
    
    def analyze_design_quality(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze design quality aspects."""
        design_analysis = {
            'inheritance_analysis': self._analyze_inheritance(tree),
            'encapsulation_analysis': self._analyze_encapsulation(tree),
            'abstraction_analysis': self._analyze_abstraction(tree),
            'dependency_analysis': self._analyze_dependencies(tree),
            'interface_analysis': self._analyze_interfaces(tree)
        }
        
        return design_analysis
    
    def calculate_maintainability_score(self, code: str, tree: ast.AST) -> Dict[str, float]:
        """Calculate maintainability score and sub-scores."""
        from ..utils.code_utils import calculate_maintainability_index
        
        mi = calculate_maintainability_index(code)
        
        # Calculate sub-scores
        readability_score = self._calculate_readability_score(code, tree)
        testability_score = self._calculate_testability_score(tree)
        modularity_score = self._calculate_modularity_score(tree)
        
        # Overall maintainability
        overall_score = (mi/100 + readability_score + testability_score + modularity_score) / 4
        
        return {
            'overall_score': overall_score,
            'maintainability_index': mi,
            'readability_score': readability_score,
            'testability_score': testability_score,
            'modularity_score': modularity_score
        }
    
    def generate_recommendations(self, code: str, tree: ast.AST) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Analyze metrics to generate recommendations
        metrics = self.calculate_code_metrics(code, tree)
        
        # Function-based recommendations
        for func_metric in metrics['function_metrics']:
            if func_metric['complexity'] > self.thresholds['cyclomatic_complexity']:
                recommendations.append({
                    'type': 'complexity_reduction',
                    'priority': 'high',
                    'description': f"Function '{func_metric['name']}' has high complexity ({func_metric['complexity']})",
                    'suggestion': 'Consider breaking down this function using Extract Method refactoring',
                    'impact': 'high'
                })
            
            if func_metric['length'] > self.thresholds['method_length']:
                recommendations.append({
                    'type': 'method_length',
                    'priority': 'medium',
                    'description': f"Function '{func_metric['name']}' is too long ({func_metric['length']} lines)",
                    'suggestion': 'Extract smaller, focused methods from this function',
                    'impact': 'medium'
                })
        
        # Class-based recommendations
        for class_metric in metrics['class_metrics']:
            if class_metric['method_count'] > self.thresholds['class_methods']:
                recommendations.append({
                    'type': 'class_size',
                    'priority': 'high',
                    'description': f"Class '{class_metric['name']}' has too many methods ({class_metric['method_count']})",
                    'suggestion': 'Consider splitting this class using Extract Class refactoring',
                    'impact': 'high'
                })
        
        # Duplication recommendations
        if metrics['duplication_score'] > 0.3:
            recommendations.append({
                'type': 'code_duplication',
                'priority': 'medium',
                'description': f"High code duplication detected ({metrics['duplication_score']:.2%})",
                'suggestion': 'Identify and extract common code into reusable methods',
                'impact': 'medium'
            })
        
        return recommendations
    
    def _find_extract_method_opportunities(self, tree: ast.AST) -> List[RefactoringOpportunity]:
        """Find Extract Method refactoring opportunities."""
        opportunities = []
        
        functions = self.ast_analyzer.find_nodes_by_type(tree, ast.FunctionDef)
        for func in functions:
            length = getattr(func, 'end_lineno', func.lineno) - func.lineno + 1
            complexity = self.ast_analyzer.calculate_complexity(func)
            
            if length > self.thresholds['method_length'] or complexity > self.thresholds['cyclomatic_complexity']:
                confidence = min(0.9, (length / self.thresholds['method_length']) * 0.4 + 
                               (complexity / self.thresholds['cyclomatic_complexity']) * 0.4 + 0.2)
                
                opportunities.append(RefactoringOpportunity(
                    type="extract_method",
                    location=f"line {func.lineno}",
                    description=f"Method '{func.name}' is long ({length} lines) or complex (complexity: {complexity})",
                    priority="high" if length > self.thresholds['method_length'] * 1.5 else "medium",
                    effort="medium",
                    confidence=confidence,
                    details={
                        'method_name': func.name,
                        'length': length,
                        'complexity': complexity,
                        'suggested_extractions': self._suggest_extraction_points(func)
                    }
                ))
        
        return opportunities
    
    def _find_extract_variable_opportunities(self, tree: ast.AST) -> List[RefactoringOpportunity]:
        """Find Extract Variable refactoring opportunities."""
        opportunities = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.BinOp, ast.BoolOp, ast.Compare)):
                if self._is_complex_expression(node):
                    opportunities.append(RefactoringOpportunity(
                        type="extract_variable",
                        location=f"line {node.lineno}",
                        description="Complex expression found that could benefit from variable extraction",
                        priority="low",
                        effort="low",
                        confidence=0.6,
                        details={
                            'expression': ast.unparse(node),
                            'complexity_score': self._calculate_expression_complexity(node)
                        }
                    ))
        
        return opportunities
    
    def _find_magic_number_opportunities(self, tree: ast.AST) -> List[RefactoringOpportunity]:
        """Find Replace Magic Numbers refactoring opportunities."""
        opportunities = []
        
        magic_numbers = self.ast_matcher.find_pattern(tree, 'magic_number')
        for match in magic_numbers:
            opportunities.append(RefactoringOpportunity(
                type="replace_magic_numbers",
                location=match['location'],
                description=f"Magic number {match['details']['value']} should be replaced with named constant",
                priority="low",
                effort="low",
                confidence=0.8,
                details=match['details']
            ))
        
        return opportunities
    
    def _find_conditional_simplification_opportunities(self, tree: ast.AST) -> List[RefactoringOpportunity]:
        """Find Simplify Conditional refactoring opportunities."""
        opportunities = []
        
        complex_conditionals = self.ast_matcher.find_pattern(tree, 'complex_conditional')
        for match in complex_conditionals:
            opportunities.append(RefactoringOpportunity(
                type="simplify_conditional",
                location=match['location'],
                description=f"Complex conditional with {match['details']['complexity']} boolean operators",
                priority="medium",
                effort="low",
                confidence=0.7,
                details=match['details']
            ))
        
        # Find deeply nested conditionals
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                nesting_depth = self._calculate_conditional_nesting(node)
                if nesting_depth > 3:
                    opportunities.append(RefactoringOpportunity(
                        type="simplify_conditional",
                        location=f"line {node.lineno}",
                        description=f"Deeply nested conditional (depth: {nesting_depth})",
                        priority="medium",
                        effort="medium",
                        confidence=0.8,
                        details={'nesting_depth': nesting_depth}
                    ))
        
        return opportunities
    
    def _find_dead_code_opportunities(self, code: str, tree: ast.AST) -> List[RefactoringOpportunity]:
        """Find Remove Dead Code refactoring opportunities."""
        opportunities = []
        
        # Find unused imports
        from ..utils.code_utils import find_unused_imports
        unused_imports = find_unused_imports(code)
        
        if unused_imports:
            opportunities.append(RefactoringOpportunity(
                type="remove_dead_code",
                location="imports",
                description=f"Found {len(unused_imports)} unused imports",
                priority="low",
                effort="low",
                confidence=0.9,
                details={'unused_imports': unused_imports}
            ))
        
        # Find unreachable code
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if self._is_unreachable_condition(node):
                    opportunities.append(RefactoringOpportunity(
                        type="remove_dead_code",
                        location=f"line {node.lineno}",
                        description="Unreachable conditional block detected",
                        priority="medium",
                        effort="low",
                        confidence=0.8,
                        details={'condition': ast.unparse(node.test)}
                    ))
        
        # Find unused variables
        unused_vars = self.ast_analyzer.find_unused_variables(tree)
        if unused_vars:
            opportunities.append(RefactoringOpportunity(
                type="remove_dead_code",
                location="variables",
                description=f"Found {len(unused_vars)} unused variables",
                priority="low",
                effort="low",
                confidence=0.7,
                details={'unused_variables': unused_vars}
            ))
        
        return opportunities
    
    def _find_extract_class_opportunities(self, tree: ast.AST) -> List[RefactoringOpportunity]:
        """Find Extract Class refactoring opportunities."""
        opportunities = []
        
        classes = self.ast_analyzer.find_nodes_by_type(tree, ast.ClassDef)
        for cls in classes:
            cls_info = self.ast_analyzer.get_class_info(cls)
            method_count = len(cls_info['methods'])
            
            if method_count > self.thresholds['class_methods']:
                # Analyze method groupings for potential extraction
                method_groups = self._analyze_method_cohesion(cls_info['methods'])
                
                for group in method_groups:
                    if len(group['methods']) >= 3:
                        opportunities.append(RefactoringOpportunity(
                            type="extract_class",
                            location=f"line {cls.lineno}",
                            description=f"Class '{cls_info['name']}' has {method_count} methods, potential extraction: {group['category']}",
                            priority="high" if method_count > self.thresholds['class_methods'] * 1.5 else "medium",
                            effort="high",
                            confidence=0.6,
                            details={
                                'class_name': cls_info['name'],
                                'method_count': method_count,
                                'extraction_candidate': group
                            }
                        ))
        
        return opportunities
    
    def _find_inline_method_opportunities(self, tree: ast.AST) -> List[RefactoringOpportunity]:
        """Find Inline Method refactoring opportunities."""
        opportunities = []
        
        functions = self.ast_analyzer.find_nodes_by_type(tree, ast.FunctionDef)
        call_graph = self.ast_analyzer.extract_method_calls_graph(tree)
        
        for func in functions:
            # Check if method is very short and only called once
            length = getattr(func, 'end_lineno', func.lineno) - func.lineno + 1
            if length <= 3:  # Very short method
                call_count = sum(1 for calls in call_graph.values() if func.name in calls)
                
                if call_count == 1:
                    opportunities.append(RefactoringOpportunity(
                        type="inline_method",
                        location=f"line {func.lineno}",
                        description=f"Short method '{func.name}' ({length} lines) called only once",
                        priority="low",
                        effort="low",
                        confidence=0.6,
                        details={
                            'method_name': func.name,
                            'length': length,
                            'call_count': call_count
                        }
                    ))
        
        return opportunities
    
    def _find_move_method_opportunities(self, tree: ast.AST) -> List[RefactoringOpportunity]:
        """Find Move Method refactoring opportunities."""
        opportunities = []
        
        classes = self.ast_analyzer.find_nodes_by_type(tree, ast.ClassDef)
        
        for cls in classes:
            cls_info = self.ast_analyzer.get_class_info(cls)
            
            for method_info in cls_info['methods']:
                # Check if method uses more external attributes than internal
                external_usage = self._count_external_attribute_usage(method_info)
                internal_usage = self._count_internal_attribute_usage(method_info, cls_info)
                
                if external_usage > internal_usage and external_usage > 2:
                    opportunities.append(RefactoringOpportunity(
                        type="move_method",
                        location=f"line {method_info['line_start']}",
                        description=f"Method '{method_info['name']}' uses more external than internal attributes",
                        priority="medium",
                        effort="medium",
                        confidence=0.5,
                        details={
                            'method_name': method_info['name'],
                            'external_usage': external_usage,
                            'internal_usage': internal_usage
                        }
                    ))
        
        return opportunities
    
    def _find_naming_issues(self, tree: ast.AST) -> List[QualityIssue]:
        """Find naming convention issues."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not self._is_valid_function_name(node.name):
                    issues.append(QualityIssue(
                        type="naming_convention",
                        severity="low",
                        location=f"line {node.lineno}",
                        description=f"Function name '{node.name}' doesn't follow Python naming conventions",
                        suggestion="Use snake_case for function names",
                        details={'name': node.name, 'type': 'function'}
                    ))
            
            elif isinstance(node, ast.ClassDef):
                if not self._is_valid_class_name(node.name):
                    issues.append(QualityIssue(
                        type="naming_convention",
                        severity="low",
                        location=f"line {node.lineno}",
                        description=f"Class name '{node.name}' doesn't follow Python naming conventions",
                        suggestion="Use PascalCase for class names",
                        details={'name': node.name, 'type': 'class'}
                    ))
        
        return issues
    
    def _find_documentation_issues(self, tree: ast.AST) -> List[QualityIssue]:
        """Find documentation issues."""
        issues = []
        
        functions = self.ast_analyzer.find_nodes_by_type(tree, ast.FunctionDef)
        classes = self.ast_analyzer.find_nodes_by_type(tree, ast.ClassDef)
        
        # Check for missing docstrings
        for func in functions:
            if not ast.get_docstring(func) and not func.name.startswith('_'):
                issues.append(QualityIssue(
                    type="missing_documentation",
                    severity="medium",
                    location=f"line {func.lineno}",
                    description=f"Public function '{func.name}' lacks documentation",
                    suggestion="Add a docstring describing the function's purpose, parameters, and return value",
                    details={'name': func.name, 'type': 'function'}
                ))
        
        for cls in classes:
            if not ast.get_docstring(cls):
                issues.append(QualityIssue(
                    type="missing_documentation",
                    severity="medium",
                    location=f"line {cls.lineno}",
                    description=f"Class '{cls.name}' lacks documentation",
                    suggestion="Add a docstring describing the class's purpose and usage",
                    details={'name': cls.name, 'type': 'class'}
                ))
        
        return issues
    
    def _find_performance_issues(self, tree: ast.AST) -> List[QualityIssue]:
        """Find potential performance issues."""
        issues = []
        
        for node in ast.walk(tree):
            # Nested loops
            if isinstance(node, (ast.For, ast.While)):
                nesting_level = self._count_loop_nesting(node)
                if nesting_level > 2:
                    issues.append(QualityIssue(
                        type="performance",
                        severity="medium",
                        location=f"line {node.lineno}",
                        description=f"Deeply nested loops (level {nesting_level}) may impact performance",
                        suggestion="Consider optimizing the algorithm or using more efficient data structures",
                        details={'nesting_level': nesting_level}
                    ))
            
            # String concatenation in loops
            elif isinstance(node, ast.For):
                if self._has_string_concatenation_in_loop(node):
                    issues.append(QualityIssue(
                        type="performance",
                        severity="medium",
                        location=f"line {node.lineno}",
                        description="String concatenation in loop can be inefficient",
                        suggestion="Use list.join() or f-strings for better performance",
                        details={'issue': 'string_concatenation_in_loop'}
                    ))
        
        return issues
    
    def _find_security_issues(self, tree: ast.AST) -> List[QualityIssue]:
        """Find potential security issues."""
        issues = []
        
        for node in ast.walk(tree):
            # Use of eval()
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'eval':
                issues.append(QualityIssue(
                    type="security",
                    severity="high",
                    location=f"line {node.lineno}",
                    description="Use of eval() can be dangerous",
                    suggestion="Consider safer alternatives like ast.literal_eval() for simple cases",
                    details={'function': 'eval'}
                ))
            
            # Use of exec()
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'exec':
                issues.append(QualityIssue(
                    type="security",
                    severity="high",
                    location=f"line {node.lineno}",
                    description="Use of exec() can be dangerous",
                    suggestion="Avoid dynamic code execution when possible",
                    details={'function': 'exec'}
                ))
        
        return issues
    
    def _calculate_duplication_score(self, code: str) -> float:
        """Calculate code duplication score."""
        lines = [line.strip() for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]
        
        if len(lines) < 2:
            return 0.0
        
        line_counts = Counter(lines)
        duplicated_lines = sum(count - 1 for count in line_counts.values() if count > 1)
        
        return duplicated_lines / len(lines)
    
    def _calculate_coupling_score(self, tree: ast.AST) -> float:
        """Calculate coupling score."""
        imports = len(self.ast_analyzer.find_nodes_by_type(tree, ast.Import)) + \
                 len(self.ast_analyzer.find_nodes_by_type(tree, ast.ImportFrom))
        classes = len(self.ast_analyzer.find_nodes_by_type(tree, ast.ClassDef))
        
        if classes == 0:
            return 0.0
        
        # Simple coupling metric: imports per class
        coupling_ratio = imports / classes
        return min(1.0, coupling_ratio / 10.0)  # Normalize
    
    def _calculate_cohesion_score(self, tree: ast.AST) -> float:
        """Calculate cohesion score."""
        classes = self.ast_analyzer.find_nodes_by_type(tree, ast.ClassDef)
        
        if not classes:
            return 1.0
        
        total_cohesion = 0.0
        for cls in classes:
            cls_info = self.ast_analyzer.get_class_info(cls)
            cohesion = self._calculate_class_cohesion(cls_info)
            total_cohesion += cohesion
        
        return total_cohesion / len(classes)
    
    def _calculate_readability_score(self, code: str, tree: ast.AST) -> float:
        """Calculate readability score."""
        readability_metrics = self.code_processor._calculate_readability_metrics(code)
        return sum(readability_metrics.values()) / len(readability_metrics)
    
    def _calculate_testability_score(self, tree: ast.AST) -> float:
        """Calculate testability score."""
        functions = self.ast_analyzer.find_nodes_by_type(tree, ast.FunctionDef)
        
        if not functions:
            return 1.0
        
        testable_functions = 0
        for func in functions:
            # Simple heuristics for testability
            func_info = self.ast_analyzer.get_function_info(func)
            
            is_testable = True
            # Functions with too many parameters are harder to test
            if len(func_info['args']['positional']) > 5:
                is_testable = False
            
            # Functions with high complexity are harder to test
            if func_info['complexity'] > 10:
                is_testable = False
            
            if is_testable:
                testable_functions += 1
        
        return testable_functions / len(functions)
    
    def _calculate_modularity_score(self, tree: ast.AST) -> float:
        """Calculate modularity score."""
        functions = len(self.ast_analyzer.find_nodes_by_type(tree, ast.FunctionDef))
        classes = len(self.ast_analyzer.find_nodes_by_type(tree, ast.ClassDef))
        
        total_code_units = functions + classes
        if total_code_units == 0:
            return 0.0
        
        # Higher ratio of classes to functions indicates better modularity
        if total_code_units < 5:
            return 0.5  # Small code base
        
        class_ratio = classes / total_code_units
        return min(1.0, class_ratio * 2)  # Normalize
    
    # Helper methods
    def _suggest_extraction_points(self, func: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Suggest points where method extraction would be beneficial."""
        suggestions = []
        
        # Look for logical blocks that could be extracted
        current_block = []
        block_start = func.lineno
        
        for stmt in func.body:
            if isinstance(stmt, (ast.If, ast.For, ast.While, ast.Try)):
                if len(current_block) > 2:
                    suggestions.append({
                        'start_line': block_start,
                        'end_line': stmt.lineno - 1,
                        'reason': 'Logical block before control structure',
                        'estimated_lines': len(current_block)
                    })
                
                current_block = []
                block_start = stmt.lineno
            else:
                current_block.append(stmt)
        
        return suggestions
    
    def _is_complex_expression(self, node: ast.AST) -> bool:
        """Check if an expression is complex enough to warrant extraction."""
        complexity_score = self._calculate_expression_complexity(node)
        return complexity_score > 3
    
    def _calculate_expression_complexity(self, node: ast.AST) -> int:
        """Calculate complexity score for an expression."""
        if isinstance(node, ast.BinOp):
            return 1 + self._calculate_expression_complexity(node.left) + \
                   self._calculate_expression_complexity(node.right)
        elif isinstance(node, ast.BoolOp):
            return len(node.values)
        elif isinstance(node, ast.Compare):
            return len(node.ops)
        elif isinstance(node, ast.Call):
            return 2  # Function calls add complexity
        else:
            return 0
    
    def _calculate_conditional_nesting(self, node: ast.If, depth: int = 0) -> int:
        """Calculate nesting depth of conditional statements."""
        max_depth = depth
        
        for stmt in node.body + node.orelse:
            if isinstance(stmt, ast.If):
                nested_depth = self._calculate_conditional_nesting(stmt, depth + 1)
                max_depth = max(max_depth, nested_depth)
        
        return max_depth
    
    def _is_unreachable_condition(self, node: ast.If) -> bool:
        """Check if a conditional is unreachable."""
        # Simple check for obviously false conditions
        if isinstance(node.test, ast.Constant):
            return not node.test.value
        
        # Check for contradictory conditions
        if isinstance(node.test, ast.Compare) and len(node.test.ops) == 1:
            if isinstance(node.test.ops[0], ast.Eq):
                left = node.test.left
                right = node.test.comparators[0]
                if (isinstance(left, ast.Constant) and isinstance(right, ast.Constant) 
                    and left.value != right.value):
                    return True
        
        return False
    
    def _analyze_method_cohesion(self, methods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze method cohesion to suggest class extractions."""
        # Simple grouping based on method name patterns
        groups = []
        
        # Group by common prefixes
        prefix_groups = defaultdict(list)
        for method in methods:
            name = method['name']
            if '_' in name:
                prefix = name.split('_')[0]
                prefix_groups[prefix].append(method)
        
        for prefix, group_methods in prefix_groups.items():
            if len(group_methods) >= 3:
                groups.append({
                    'category': f"{prefix}_operations",
                    'methods': [m['name'] for m in group_methods],
                    'reason': f"Methods with common prefix '{prefix}'"
                })
        
        return groups
    
    def _count_external_attribute_usage(self, method_info: Dict[str, Any]) -> int:
        """Count usage of external attributes in a method."""
        # This would require more detailed AST analysis
        # For now, return a placeholder
        return 0
    
    def _count_internal_attribute_usage(self, method_info: Dict[str, Any], class_info: Dict[str, Any]) -> int:
        """Count usage of internal class attributes in a method."""
        # This would require more detailed AST analysis
        # For now, return a placeholder
        return 1
    
    def _is_valid_function_name(self, name: str) -> bool:
        """Check if function name follows Python conventions."""
        return name.islower() and (name.isalnum() or '_' in name)
    
    def _is_valid_class_name(self, name: str) -> bool:
        """Check if class name follows Python conventions."""
        return name[0].isupper() and name.replace('_', '').isalnum()
    
    def _count_loop_nesting(self, node: ast.AST, depth: int = 0) -> int:
        """Count loop nesting depth."""
        max_depth = depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                nested_depth = self._count_loop_nesting(child, depth + 1)
                max_depth = max(max_depth, nested_depth)
        
        return max_depth
    
    def _has_string_concatenation_in_loop(self, loop_node: ast.AST) -> bool:
        """Check if loop contains string concatenation."""
        for node in ast.walk(loop_node):
            if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
                # Check if it's string concatenation
                return True
        return False
    
    def _calculate_class_cohesion(self, class_info: Dict[str, Any]) -> float:
        """Calculate cohesion for a single class."""
        methods = class_info['methods']
        attributes = class_info['attributes']
        
        if not methods or not (attributes['class_variables'] or attributes['instance_variables']):
            return 1.0
        
        # Simple cohesion metric based on method-attribute relationships
        total_attributes = len(attributes['class_variables']) + len(attributes['instance_variables'])
        
        # This would require detailed analysis of which methods use which attributes
        # For now, return a reasonable estimate
        return 0.7
    
    def _get_smell_suggestion(self, smell_type: str) -> str:
        """Get suggestion for fixing a code smell."""
        suggestions = {
            'long_method': 'Break down this method using Extract Method refactoring',
            'large_class': 'Consider splitting this class using Extract Class refactoring',
            'magic_number': 'Replace magic number with a named constant',
            'duplicate_code': 'Extract duplicate code into a reusable method',
            'complex_conditional': 'Simplify conditional logic or extract to a method'
        }
        return suggestions.get(smell_type, 'Consider refactoring to improve code quality')
    
    def _analyze_inheritance(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze inheritance structure."""
        classes = self.ast_analyzer.find_nodes_by_type(tree, ast.ClassDef)
        
        inheritance_info = {
            'total_classes': len(classes),
            'classes_with_inheritance': 0,
            'max_inheritance_depth': 0,
            'inheritance_tree': {}
        }
        
        for cls in classes:
            if cls.bases:
                inheritance_info['classes_with_inheritance'] += 1
                inheritance_info['inheritance_tree'][cls.name] = [ast.unparse(base) for base in cls.bases]
        
        return inheritance_info
    
    def _analyze_encapsulation(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze encapsulation quality."""
        classes = self.ast_analyzer.find_nodes_by_type(tree, ast.ClassDef)
        
        encapsulation_info = {
            'total_classes': len(classes),
            'classes_with_private_methods': 0,
            'classes_with_properties': 0
        }
        
        for cls in classes:
            has_private = any(method.name.startswith('_') for method in cls.body 
                            if isinstance(method, ast.FunctionDef))
            if has_private:
                encapsulation_info['classes_with_private_methods'] += 1
        
        return encapsulation_info
    
    def _analyze_abstraction(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze abstraction level."""
        functions = self.ast_analyzer.find_nodes_by_type(tree, ast.FunctionDef)
        
        abstraction_info = {
            'total_functions': len(functions),
            'abstract_methods': 0,
            'average_abstraction_level': 0.5  # Placeholder
        }
        
        return abstraction_info
    
    def _analyze_dependencies(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze dependency structure."""
        return self.ast_analyzer.get_dependencies(tree)
    
    def _analyze_interfaces(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze interface design."""
        functions = self.ast_analyzer.find_nodes_by_type(tree, ast.FunctionDef)
        
        interface_info = {
            'public_methods': 0,
            'private_methods': 0,
            'methods_with_docstrings': 0
        }
        
        for func in functions:
            if func.name.startswith('_'):
                interface_info['private_methods'] += 1
            else:
                interface_info['public_methods'] += 1
            
            if ast.get_docstring(func):
                interface_info['methods_with_docstrings'] += 1
        
        return interface_info