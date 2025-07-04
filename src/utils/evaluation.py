# src/utils/evaluation.py
import ast
import sys
import subprocess
import tempfile
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
from collections import defaultdict
import difflib
import re

logger = logging.getLogger(__name__)

class RefactoringEvaluator:
    """Comprehensive evaluator for code refactoring results."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.metrics = [
            "syntax_correctness",
            "semantic_equivalence", 
            "code_quality_improvement",
            "readability_improvement",
            "maintainability_improvement",
            "performance_impact",
            "test_preservation"
        ]
    
    def evaluate_refactoring(
        self,
        original_code: str,
        refactored_code: str,
        test_cases: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of refactoring results.
        
        Args:
            original_code: Original source code
            refactored_code: Refactored source code
            test_cases: Optional test cases to verify behavior preservation
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        
        logger.info("Starting comprehensive refactoring evaluation")
        
        results = {
            "syntax_correctness": self.evaluate_syntax_correctness(refactored_code),
            "semantic_equivalence": self.evaluate_semantic_equivalence(
                original_code, refactored_code
            ),
            "code_quality": self.evaluate_code_quality_improvement(
                original_code, refactored_code
            ),
            "readability": self.evaluate_readability_improvement(
                original_code, refactored_code
            ),
            "maintainability": self.evaluate_maintainability_improvement(
                original_code, refactored_code
            ),
            "performance": self.evaluate_performance_impact(
                original_code, refactored_code
            ),
            "overall_score": 0.0
        }
        
        # Evaluate test preservation if test cases provided
        if test_cases:
            results["test_preservation"] = self.evaluate_test_preservation(
                original_code, refactored_code, test_cases
            )
        
        # Calculate overall score
        results["overall_score"] = self._calculate_overall_score(results)
        
        logger.info("Refactoring evaluation completed")
        return results
    
    def evaluate_syntax_correctness(self, code: str) -> Dict[str, Any]:
        """Evaluate if the refactored code is syntactically correct."""
        
        try:
            ast.parse(code)
            return {
                "is_valid": True,
                "error": None,
                "score": 1.0
            }
        except SyntaxError as e:
            return {
                "is_valid": False,
                "error": str(e),
                "score": 0.0
            }
        except Exception as e:
            return {
                "is_valid": False,
                "error": f"Unexpected error: {str(e)}",
                "score": 0.0
            }
    
    def evaluate_semantic_equivalence(
        self,
        original_code: str,
        refactored_code: str
    ) -> Dict[str, Any]:
        """Evaluate if refactored code preserves semantic meaning."""
        
        # AST-based comparison
        ast_similarity = self._compare_ast_structure(original_code, refactored_code)
        
        # Control flow comparison
        control_flow_similarity = self._compare_control_flow(original_code, refactored_code)
        
        # Variable usage comparison
        variable_similarity = self._compare_variable_usage(original_code, refactored_code)
        
        # Function signature comparison
        function_similarity = self._compare_function_signatures(original_code, refactored_code)
        
        # Overall semantic similarity
        semantic_score = (
            0.3 * ast_similarity +
            0.3 * control_flow_similarity +
            0.2 * variable_similarity +
            0.2 * function_similarity
        )
        
        return {
            "semantic_score": semantic_score,
            "ast_similarity": ast_similarity,
            "control_flow_similarity": control_flow_similarity,
            "variable_similarity": variable_similarity,
            "function_similarity": function_similarity,
            "is_equivalent": semantic_score > 0.8
        }
    
    def evaluate_code_quality_improvement(
        self,
        original_code: str,
        refactored_code: str
    ) -> Dict[str, Any]:
        """Evaluate code quality improvements."""
        
        original_metrics = self._calculate_code_metrics(original_code)
        refactored_metrics = self._calculate_code_metrics(refactored_code)
        
        improvements = {}
        for metric in original_metrics:
            original_val = original_metrics[metric]
            refactored_val = refactored_metrics[metric]
            
            if original_val > 0:
                improvement = (refactored_val - original_val) / original_val
            else:
                improvement = 0.0
            
            improvements[f"{metric}_improvement"] = improvement
        
        # Calculate overall quality improvement
        quality_metrics = [
            "maintainability_index",
            "cyclomatic_complexity",
            "lines_of_code",
            "number_of_methods"
        ]
        
        quality_improvements = []
        for metric in quality_metrics:
            if f"{metric}_improvement" in improvements:
                # For complexity and LOC, negative improvement is good
                if metric in ["cyclomatic_complexity", "lines_of_code"]:
                    quality_improvements.append(-improvements[f"{metric}_improvement"])
                else:
                    quality_improvements.append(improvements[f"{metric}_improvement"])
        
        overall_improvement = np.mean(quality_improvements) if quality_improvements else 0.0
        
        return {
            "original_metrics": original_metrics,
            "refactored_metrics": refactored_metrics,
            "improvements": improvements,
            "overall_improvement": overall_improvement,
            "quality_score": max(0.0, min(1.0, 0.5 + overall_improvement))
        }
    
    def evaluate_readability_improvement(
        self,
        original_code: str,
        refactored_code: str
    ) -> Dict[str, Any]:
        """Evaluate readability improvements."""
        
        original_readability = self._calculate_readability_metrics(original_code)
        refactored_readability = self._calculate_readability_metrics(refactored_code)
        
        improvements = {}
        for metric in original_readability:
            original_val = original_readability[metric]
            refactored_val = refactored_readability[metric]
            
            if original_val > 0:
                improvement = (refactored_val - original_val) / original_val
            else:
                improvement = 0.0
            
            improvements[f"{metric}_improvement"] = improvement
        
        # Overall readability score
        readability_score = np.mean(list(refactored_readability.values()))
        
        return {
            "original_readability": original_readability,
            "refactored_readability": refactored_readability,
            "improvements": improvements,
            "readability_score": readability_score
        }
    
    def evaluate_maintainability_improvement(
        self,
        original_code: str,
        refactored_code: str
    ) -> Dict[str, Any]:
        """Evaluate maintainability improvements."""
        
        original_maintainability = self._calculate_maintainability_metrics(original_code)
        refactored_maintainability = self._calculate_maintainability_metrics(refactored_code)
        
        improvements = {}
        for metric in original_maintainability:
            original_val = original_maintainability[metric]
            refactored_val = refactored_maintainability[metric]
            
            if original_val > 0:
                improvement = (refactored_val - original_val) / original_val
            else:
                improvement = 0.0
            
            improvements[f"{metric}_improvement"] = improvement
        
        # Overall maintainability score
        maintainability_score = np.mean(list(refactored_maintainability.values()))
        
        return {
            "original_maintainability": original_maintainability,
            "refactored_maintainability": refactored_maintainability,
            "improvements": improvements,
            "maintainability_score": maintainability_score
        }
    
    def evaluate_performance_impact(
        self,
        original_code: str,
        refactored_code: str
    ) -> Dict[str, Any]:
        """Evaluate performance impact of refactoring."""
        
        try:
            # Static analysis for performance indicators
            original_performance = self._analyze_performance_indicators(original_code)
            refactored_performance = self._analyze_performance_indicators(refactored_code)
            
            # Calculate performance score based on static analysis
            performance_improvements = {}
            for metric in original_performance:
                original_val = original_performance[metric]
                refactored_val = refactored_performance[metric]
                
                if original_val > 0:
                    improvement = (original_val - refactored_val) / original_val
                else:
                    improvement = 0.0
                
                performance_improvements[f"{metric}_improvement"] = improvement
            
            overall_performance_score = np.mean(list(performance_improvements.values()))
            
            return {
                "original_performance": original_performance,
                "refactored_performance": refactored_performance,
                "improvements": performance_improvements,
                "performance_score": max(0.0, min(1.0, 0.5 + overall_performance_score))
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating performance impact: {e}")
            return {
                "performance_score": 0.5,
                "error": str(e)
            }
    
    def evaluate_test_preservation(
        self,
        original_code: str,
        refactored_code: str,
        test_cases: List[str]
    ) -> Dict[str, Any]:
        """Evaluate if refactoring preserves test behavior."""
        
        results = {
            "total_tests": len(test_cases),
            "passed_original": 0,
            "passed_refactored": 0,
            "preservation_rate": 0.0,
            "test_results": []
        }
        
        for i, test_case in enumerate(test_cases):
            try:
                # Run test on original code
                original_result = self._run_test_case(original_code, test_case)
                
                # Run test on refactored code
                refactored_result = self._run_test_case(refactored_code, test_case)
                
                # Check if results match
                preservation = original_result == refactored_result
                
                if original_result:
                    results["passed_original"] += 1
                if refactored_result:
                    results["passed_refactored"] += 1
                
                results["test_results"].append({
                    "test_id": i,
                    "original_passed": original_result,
                    "refactored_passed": refactored_result,
                    "preserved": preservation
                })
                
            except Exception as e:
                logger.warning(f"Error running test case {i}: {e}")
                results["test_results"].append({
                    "test_id": i,
                    "error": str(e),
                    "preserved": False
                })
        
        # Calculate preservation rate
        preserved_tests = sum(1 for result in results["test_results"] if result.get("preserved", False))
        results["preservation_rate"] = preserved_tests / len(test_cases) if test_cases else 0.0
        
        return results
    
    def _compare_ast_structure(self, code1: str, code2: str) -> float:
        """Compare AST structures of two code snippets."""
        
        try:
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)
            
            # Extract structural features
            features1 = self._extract_ast_features(tree1)
            features2 = self._extract_ast_features(tree2)
            
            # Calculate similarity
            similarity = self._calculate_feature_similarity(features1, features2)
            return similarity
            
        except Exception as e:
            logger.warning(f"Error comparing AST structures: {e}")
            return 0.0
    
    def _extract_ast_features(self, tree: ast.AST) -> Dict[str, int]:
        """Extract features from AST."""
        
        features = defaultdict(int)
        
        for node in ast.walk(tree):
            node_type = type(node).__name__
            features[f"node_{node_type}"] += 1
            
            # Special handling for certain node types
            if isinstance(node, ast.FunctionDef):
                features["function_count"] += 1
                features["function_args"] += len(node.args.args)
            elif isinstance(node, ast.ClassDef):
                features["class_count"] += 1
            elif isinstance(node, (ast.If, ast.For, ast.While)):
                features["control_flow"] += 1
        
        return dict(features)
    
    def _calculate_feature_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between feature dictionaries."""
        
        all_features = set(features1.keys()) | set(features2.keys())
        
        if not all_features:
            return 1.0
        
        similarities = []
        for feature in all_features:
            val1 = features1.get(feature, 0)
            val2 = features2.get(feature, 0)
            
            if val1 == 0 and val2 == 0:
                similarity = 1.0
            elif val1 == 0 or val2 == 0:
                similarity = 0.0
            else:
                similarity = min(val1, val2) / max(val1, val2)
            
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def _compare_control_flow(self, code1: str, code2: str) -> float:
        """Compare control flow structures."""
        
        try:
            flow1 = self._extract_control_flow(code1)
            flow2 = self._extract_control_flow(code2)
            
            return self._calculate_feature_similarity(flow1, flow2)
            
        except Exception as e:
            logger.warning(f"Error comparing control flow: {e}")
            return 0.0
    
    def _extract_control_flow(self, code: str) -> Dict[str, int]:
        """Extract control flow features."""
        
        tree = ast.parse(code)
        flow_features = defaultdict(int)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                flow_features["if_statements"] += 1
            elif isinstance(node, ast.For):
                flow_features["for_loops"] += 1
            elif isinstance(node, ast.While):
                flow_features["while_loops"] += 1
            elif isinstance(node, ast.Try):
                flow_features["try_blocks"] += 1
            elif isinstance(node, ast.With):
                flow_features["with_statements"] += 1
        
        return dict(flow_features)
    
    def _compare_variable_usage(self, code1: str, code2: str) -> float:
        """Compare variable usage patterns."""
        
        try:
            vars1 = self._extract_variable_info(code1)
            vars2 = self._extract_variable_info(code2)
            
            return self._calculate_feature_similarity(vars1, vars2)
            
        except Exception as e:
            logger.warning(f"Error comparing variable usage: {e}")
            return 0.0
    
    def _extract_variable_info(self, code: str) -> Dict[str, int]:
        """Extract variable usage information."""
        
        tree = ast.parse(code)
        var_info = defaultdict(int)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    var_info["assignments"] += 1
                elif isinstance(node.ctx, ast.Load):
                    var_info["reads"] += 1
        
        return dict(var_info)
    
    def _compare_function_signatures(self, code1: str, code2: str) -> float:
        """Compare function signatures."""
        
        try:
            sigs1 = self._extract_function_signatures(code1)
            sigs2 = self._extract_function_signatures(code2)
            
            # Compare function names and argument counts
            if not sigs1 and not sigs2:
                return 1.0
            
            common_functions = set(sigs1.keys()) & set(sigs2.keys())
            
            if not common_functions:
                return 0.0
            
            similarities = []
            for func_name in common_functions:
                sig1 = sigs1[func_name]
                sig2 = sigs2[func_name]
                
                # Compare argument counts
                if sig1["arg_count"] == sig2["arg_count"]:
                    similarities.append(1.0)
                else:
                    similarity = min(sig1["arg_count"], sig2["arg_count"]) / max(sig1["arg_count"], sig2["arg_count"])
                    similarities.append(similarity)
            
            return np.mean(similarities)
            
        except Exception as e:
            logger.warning(f"Error comparing function signatures: {e}")
            return 0.0
    
    def _extract_function_signatures(self, code: str) -> Dict[str, Dict]:
        """Extract function signature information."""
        
        tree = ast.parse(code)
        signatures = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                signatures[node.name] = {
                    "arg_count": len(node.args.args),
                    "has_varargs": node.args.vararg is not None,
                    "has_kwargs": node.args.kwarg is not None
                }
        
        return signatures
    
    def _calculate_code_metrics(self, code: str) -> Dict[str, float]:
        """Calculate various code metrics."""
        
        try:
            tree = ast.parse(code)
            
            metrics = {
                "lines_of_code": len([line for line in code.split('\n') if line.strip()]),
                "cyclomatic_complexity": self._calculate_cyclomatic_complexity(tree),
                "number_of_methods": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                "number_of_classes": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                "depth_of_inheritance": self._calculate_inheritance_depth(tree),
                "maintainability_index": self._calculate_maintainability_index(code, tree)
            }
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating code metrics: {e}")
            return {
                "lines_of_code": 0,
                "cyclomatic_complexity": 1,
                "number_of_methods": 0,
                "number_of_classes": 0,
                "depth_of_inheritance": 0,
                "maintainability_index": 0.5
            }
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                complexity += 1
        
        return complexity
    
    def _calculate_inheritance_depth(self, tree: ast.AST) -> int:
        """Calculate maximum inheritance depth."""
        
        max_depth = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                depth = len(node.bases)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_maintainability_index(self, code: str, tree: ast.AST) -> float:
        """Calculate maintainability index."""
        
        try:
            loc = len([line for line in code.split('\n') if line.strip()])
            complexity = self._calculate_cyclomatic_complexity(tree)
            
            # Simplified maintainability index
            if loc > 0:
                mi = max(0, (171 - 5.2 * np.log(loc) - 0.23 * complexity) / 171)
            else:
                mi = 0.5
            
            return mi
            
        except Exception:
            return 0.5
    
    def _calculate_readability_metrics(self, code: str) -> Dict[str, float]:
        """Calculate readability metrics."""
        
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return {
                "average_line_length": 0.0,
                "comment_ratio": 0.0,
                "naming_quality": 0.0,
                "indentation_consistency": 0.0
            }
        
        # Average line length
        avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
        line_length_score = max(0.0, 1.0 - abs(avg_line_length - 80) / 80)
        
        # Comment ratio
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        comment_ratio = comment_lines / len(non_empty_lines)
        comment_score = min(1.0, comment_ratio / 0.2)  # Target 20% comments
        
        # Naming quality (simplified)
        naming_score = self._evaluate_naming_quality(code)
        
        # Indentation consistency
        indentation_score = self._evaluate_indentation_consistency(lines)
        
        return {
            "average_line_length": line_length_score,
            "comment_ratio": comment_score,
            "naming_quality": naming_score,
            "indentation_consistency": indentation_score
        }
    
    def _calculate_maintainability_metrics(self, code: str) -> Dict[str, float]:
        """Calculate maintainability metrics."""
        
        try:
            tree = ast.parse(code)
            
            # Method length
            method_lengths = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    method_length = node.end_lineno - node.lineno + 1
                    method_lengths.append(method_length)
            
            avg_method_length = np.mean(method_lengths) if method_lengths else 0
            method_length_score = max(0.0, 1.0 - max(0, avg_method_length - 20) / 30)
            
            # Class cohesion (simplified)
            cohesion_score = self._calculate_class_cohesion(tree)
            
            # Coupling (simplified)
            coupling_score = self._calculate_coupling_metric(tree)
            
            # Code duplication
            duplication_score = self._calculate_duplication_metric(code)
            
            return {
                "method_length": method_length_score,
                "class_cohesion": cohesion_score,
                "coupling": 1.0 - coupling_score,  # Lower coupling is better
                "code_duplication": 1.0 - duplication_score  # Lower duplication is better
            }
            
        except Exception as e:
            logger.warning(f"Error calculating maintainability metrics: {e}")
            return {
                "method_length": 0.5,
                "class_cohesion": 0.5,
                "coupling": 0.5,
                "code_duplication": 0.5
            }
    
    def _analyze_performance_indicators(self, code: str) -> Dict[str, int]:
        """Analyze static performance indicators."""
        
        try:
            tree = ast.parse(code)
            
            indicators = {
                "nested_loops": 0,
                "recursive_calls": 0,
                "list_comprehensions": 0,
                "lambda_functions": 0,
                "string_concatenations": 0
            }
            
            # Count nested loops
            def count_nested_loops(node, depth=0):
                if isinstance(node, (ast.For, ast.While)):
                    depth += 1
                    indicators["nested_loops"] = max(indicators["nested_loops"], depth)
                
                for child in ast.iter_child_nodes(node):
                    count_nested_loops(child, depth)
            
            count_nested_loops(tree)
            
            # Count other performance indicators
            for node in ast.walk(tree):
                if isinstance(node, ast.ListComp):
                    indicators["list_comprehensions"] += 1
                elif isinstance(node, ast.Lambda):
                    indicators["lambda_functions"] += 1
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    # Check for string concatenation
                    if self._is_string_operation(node):
                        indicators["string_concatenations"] += 1
            
            return indicators
            
        except Exception as e:
            logger.warning(f"Error analyzing performance indicators: {e}")
            return {
                "nested_loops": 0,
                "recursive_calls": 0,
                "list_comprehensions": 0,
                "lambda_functions": 0,
                "string_concatenations": 0
            }
    
    def _run_test_case(self, code: str, test_case: str) -> bool:
        """Run a test case on the given code."""
        
        try:
            # Create temporary file with code and test
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code + '\n\n' + test_case)
                temp_file = f.name
            
            # Run the test
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up
            os.unlink(temp_file)
            
            # Return True if test passed (no exception)
            return result.returncode == 0
            
        except Exception as e:
            logger.warning(f"Error running test case: {e}")
            return False
    
    def _evaluate_naming_quality(self, code: str) -> float:
        """Evaluate naming quality."""
        
        try:
            tree = ast.parse(code)
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
            
        except Exception:
            return 0.5
    
    def _is_good_function_name(self, name: str) -> bool:
        """Check if function name follows good practices."""
        # Snake case, descriptive length
        return (name.islower() and 
                len(name) >= 3 and 
                not name.startswith('_') and
                '_' in name or len(name) <= 15)
    
    def _is_good_class_name(self, name: str) -> bool:
        """Check if class name follows good practices."""
        # PascalCase, descriptive
        return (name[0].isupper() and 
                len(name) >= 3 and 
                name.isalnum())
    
    def _is_good_variable_name(self, name: str) -> bool:
        """Check if variable name follows good practices."""
        # Not single letter (except common loop vars), descriptive
        common_short_names = {'i', 'j', 'k', 'x', 'y', 'z'}
        return (len(name) > 1 or name in common_short_names) and name.islower()
    
    def _evaluate_indentation_consistency(self, lines: List[str]) -> float:
        """Evaluate indentation consistency."""
        
        indentations = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indentations.append(indent)
        
        if not indentations:
            return 1.0
        
        # Check if all indentations are multiples of 4
        consistent = all(indent % 4 == 0 for indent in indentations)
        return 1.0 if consistent else 0.5
    
    def _calculate_class_cohesion(self, tree: ast.AST) -> float:
        """Calculate class cohesion metric."""
        
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        if not classes:
            return 1.0
        
        total_cohesion = 0.0
        
        for class_node in classes:
            methods = [node for node in class_node.body if isinstance(node, ast.FunctionDef)]
            
            if len(methods) <= 1:
                total_cohesion += 1.0
                continue
            
            # Simplified cohesion: methods that use instance variables
            method_variable_usage = 0
            for method in methods:
                uses_instance_vars = any(
                    isinstance(node, ast.Attribute) and
                    isinstance(node.value, ast.Name) and
                    node.value.id == 'self'
                    for node in ast.walk(method)
                )
                if uses_instance_vars:
                    method_variable_usage += 1
            
            cohesion = method_variable_usage / len(methods)
            total_cohesion += cohesion
        
        return total_cohesion / len(classes)
    
    def _calculate_coupling_metric(self, tree: ast.AST) -> float:
        """Calculate coupling metric."""
        
        imports = len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))])
        classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        
        if classes == 0:
            return 0.0
        
        # Simple coupling metric: imports per class
        coupling = imports / classes
        return min(1.0, coupling / 5.0)  # Normalize to 0-1
    
    def _calculate_duplication_metric(self, code: str) -> float:
        """Calculate code duplication metric."""
        
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return 0.0
        
        unique_lines = len(set(lines))
        duplication_ratio = 1.0 - (unique_lines / len(lines))
        
        return duplication_ratio
    
    def _is_string_operation(self, node: ast.BinOp) -> bool:
        """Check if binary operation is likely string concatenation."""
        # Simplified check - in practice, would need more sophisticated analysis
        return isinstance(node.op, ast.Add)
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall refactoring score."""
        
        scores = []
        weights = {
            "syntax_correctness": 0.3,
            "semantic_equivalence": 0.2,
            "code_quality": 0.2,
            "readability": 0.15,
            "maintainability": 0.15
        }
        
        for metric, weight in weights.items():
            if metric in results:
                if metric == "syntax_correctness":
                    score = results[metric]["score"]
                elif metric == "semantic_equivalence":
                    score = results[metric]["semantic_score"]
                elif metric in ["code_quality", "readability", "maintainability"]:
                    score = results[metric].get("overall_improvement", 0.0)
                    score = max(0.0, min(1.0, 0.5 + score))  # Normalize to 0-1
                else:
                    score = 0.5
                
                scores.append(weight * score)
        
        return sum(scores)
    
    def generate_evaluation_report(
        self,
        results: Dict[str, Any],
        output_file: str = None
    ) -> str:
        """Generate a comprehensive evaluation report."""
        
        report = []
        report.append("=" * 60)
        report.append("CODE REFACTORING EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall score
        overall_score = results.get("overall_score", 0.0)
        report.append(f"Overall Refactoring Score: {overall_score:.3f}/1.000")
        report.append("")
        
        # Syntax correctness
        if "syntax_correctness" in results:
            syntax = results["syntax_correctness"]
            status = "PASS" if syntax["is_valid"] else "FAIL"
            report.append(f"Syntax Correctness: {status}")
            if not syntax["is_valid"]:
                report.append(f"  Error: {syntax['error']}")
            report.append("")
        
        # Semantic equivalence
        if "semantic_equivalence" in results:
            semantic = results["semantic_equivalence"]
            report.append(f"Semantic Equivalence: {semantic['semantic_score']:.3f}")
            report.append(f"  AST Similarity: {semantic['ast_similarity']:.3f}")
            report.append(f"  Control Flow Similarity: {semantic['control_flow_similarity']:.3f}")
            report.append(f"  Variable Usage Similarity: {semantic['variable_similarity']:.3f}")
            report.append("")
        
        # Code quality
        if "code_quality" in results:
            quality = results["code_quality"]
            report.append(f"Code Quality Improvement: {quality['overall_improvement']:.3f}")
            report.append("  Detailed improvements:")
            for metric, improvement in quality["improvements"].items():
                report.append(f"    {metric}: {improvement:.3f}")
            report.append("")
        
        # Readability
        if "readability" in results:
            readability = results["readability"]
            report.append(f"Readability Score: {readability['readability_score']:.3f}")
            report.append("")
        
        # Maintainability
        if "maintainability" in results:
            maintainability = results["maintainability"]
            report.append(f"Maintainability Score: {maintainability['maintainability_score']:.3f}")
            report.append("")
        
        # Performance
        if "performance" in results:
            performance = results["performance"]
            report.append(f"Performance Score: {performance['performance_score']:.3f}")
            report.append("")
        
        # Test preservation
        if "test_preservation" in results:
            tests = results["test_preservation"]
            report.append(f"Test Preservation Rate: {tests['preservation_rate']:.3f}")
            report.append(f"  Total tests: {tests['total_tests']}")
            report.append(f"  Passed on original: {tests['passed_original']}")
            report.append(f"  Passed on refactored: {tests['passed_refactored']}")
            report.append("")
        
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {output_file}")
        
        return report_text