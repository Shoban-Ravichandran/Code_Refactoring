import random
import ast
import os
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from pathlib import Path

@dataclass
class CodePattern:
    """Represents a code pattern with before and after refactoring."""
    original_code: str
    refactored_code: str
    pattern_type: str
    description: str
    quality_improvement: Dict[str, float]
    complexity_reduction: float

class SyntheticDataGenerator:
    """Generates synthetic code refactoring datasets."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.patterns = self._initialize_patterns()
        self.variable_names = self._load_common_names()
        self.method_names = self._load_method_names()
        self.class_names = self._load_class_names()
        
    def _initialize_patterns(self) -> List[str]:
        """Initialize refactoring patterns."""
        return [
            "extract_method", "inline_method", "extract_variable", "inline_variable",
            "move_method", "rename_method", "extract_class", "inline_class",
            "replace_conditional_with_polymorphism", "replace_magic_numbers",
            "remove_dead_code", "simplify_conditional"
        ]
    
    def _load_common_names(self) -> List[str]:
        """Load common variable names."""
        return [
            "data", "result", "value", "item", "element", "node", "temp", "buffer",
            "index", "count", "size", "length", "total", "sum", "average", "max",
            "min", "current", "previous", "next", "first", "last", "input", "output",
            "source", "target", "destination", "config", "settings", "options",
            "parameters", "arguments", "response", "request", "message", "error"
        ]
    
    def _load_method_names(self) -> List[str]:
        """Load common method names."""
        return [
            "calculate", "process", "validate", "transform", "convert", "parse",
            "format", "serialize", "deserialize", "execute", "run", "start", "stop",
            "initialize", "finalize", "create", "destroy", "build", "update", "delete",
            "find", "search", "filter", "sort", "compare", "merge", "split", "join",
            "encode", "decode", "encrypt", "decrypt", "compress", "decompress"
        ]
    
    def _load_class_names(self) -> List[str]:
        """Load common class names."""
        return [
            "Manager", "Handler", "Controller", "Service", "Repository", "Factory",
            "Builder", "Validator", "Transformer", "Processor", "Analyzer", "Parser",
            "Formatter", "Serializer", "Deserializer", "Executor", "Runner", "Worker",
            "Client", "Server", "Provider", "Consumer", "Producer", "Observer",
            "Listener", "Strategy", "Command", "State", "Context", "Adapter"
        ]
    
    def generate_extract_method_pattern(self) -> CodePattern:
        """Generate extract method refactoring pattern."""
        method_name = random.choice(self.method_names)
        var_names = random.sample(self.variable_names, k=random.randint(3, 6))
        
        # Generate original code with long method
        original_code = f"""def {method_name}(self, {', '.join(var_names[:2])}):
    # Long method that needs refactoring
    {var_names[2]} = {var_names[0]} + {var_names[1]}
    if {var_names[2]} > 100:
        {var_names[3]} = {var_names[2]} * 2
        {var_names[4]} = {var_names[3]} / 3
        if {var_names[4]} > 50:
            {var_names[5]} = {var_names[4]} - 10
            return {var_names[5]}
        else:
            {var_names[5]} = {var_names[4]} + 20
            return {var_names[5]}
    else:
        {var_names[3]} = {var_names[2]} * 3
        return {var_names[3]}"""
        
        # Generate refactored code
        extracted_method = f"_calculate_{random.choice(['result', 'value', 'total'])}"
        refactored_code = f"""def {method_name}(self, {', '.join(var_names[:2])}):
    {var_names[2]} = {var_names[0]} + {var_names[1]}
    if {var_names[2]} > 100:
        return self.{extracted_method}({var_names[2]})
    else:
        return {var_names[2]} * 3

def {extracted_method}(self, {var_names[2]}):
    {var_names[3]} = {var_names[2]} * 2
    {var_names[4]} = {var_names[3]} / 3
    if {var_names[4]} > 50:
        return {var_names[4]} - 10
    else:
        return {var_names[4]} + 20"""
        
        return CodePattern(
            original_code=original_code,
            refactored_code=refactored_code,
            pattern_type="extract_method",
            description="Extract complex logic into separate method",
            quality_improvement={
                "readability": 0.8,
                "maintainability": 0.7,
                "reusability": 0.6
            },
            complexity_reduction=0.4
        )
    
    def generate_extract_variable_pattern(self) -> CodePattern:
        """Generate extract variable refactoring pattern."""
        var_names = random.sample(self.variable_names, k=random.randint(4, 7))
        
        # Generate original code with complex expressions
        original_code = f"""def calculate_{random.choice(['price', 'total', 'score'])}(self, {var_names[0]}, {var_names[1]}):
    return ({var_names[0]} * 1.2 + {var_names[1]} * 0.8) * (1 + 0.15 if {var_names[0]} > 100 else 1 + 0.05)"""
        
        # Generate refactored code
        refactored_code = f"""def calculate_{random.choice(['price', 'total', 'score'])}(self, {var_names[0]}, {var_names[1]}):
    {var_names[2]} = {var_names[0]} * 1.2 + {var_names[1]} * 0.8
    {var_names[3]} = 0.15 if {var_names[0]} > 100 else 0.05
    {var_names[4]} = 1 + {var_names[3]}
    return {var_names[2]} * {var_names[4]}"""
        
        return CodePattern(
            original_code=original_code,
            refactored_code=refactored_code,
            pattern_type="extract_variable",
            description="Extract complex expressions into named variables",
            quality_improvement={
                "readability": 0.9,
                "maintainability": 0.6,
                "debuggability": 0.8
            },
            complexity_reduction=0.3
        )
    
    def generate_replace_magic_numbers_pattern(self) -> CodePattern:
        """Generate replace magic numbers refactoring pattern."""
        var_names = random.sample(self.variable_names, k=3)
        magic_numbers = [42, 100, 365, 1000, 0.15, 0.8, 3.14159]
        chosen_numbers = random.sample(magic_numbers, k=2)
        
        # Generate original code with magic numbers
        original_code = f"""def process_{random.choice(['data', 'request', 'input'])}(self, {var_names[0]}):
    if {var_names[0]} > {chosen_numbers[0]}:
        {var_names[1]} = {var_names[0]} * {chosen_numbers[1]}
        return {var_names[1]}
    return {var_names[0]}"""
        
        # Generate refactored code
        const_names = [
            f"{'_'.join(random.choice(['MAX', 'MIN', 'DEFAULT']).split())}_{'_'.join(random.choice(['THRESHOLD', 'LIMIT', 'VALUE']).split())}",
            f"{'_'.join(random.choice(['MULTIPLIER', 'FACTOR', 'RATE']).split())}_{'_'.join(random.choice(['VALUE', 'CONSTANT']).split())}"
        ]
        
        refactored_code = f"""# Constants
{const_names[0]} = {chosen_numbers[0]}
{const_names[1]} = {chosen_numbers[1]}

def process_{random.choice(['data', 'request', 'input'])}(self, {var_names[0]}):
    if {var_names[0]} > {const_names[0]}:
        {var_names[1]} = {var_names[0]} * {const_names[1]}
        return {var_names[1]}
    return {var_names[0]}"""
        
        return CodePattern(
            original_code=original_code,
            refactored_code=refactored_code,
            pattern_type="replace_magic_numbers",
            description="Replace magic numbers with named constants",
            quality_improvement={
                "readability": 0.7,
                "maintainability": 0.8,
                "configuration": 0.9
            },
            complexity_reduction=0.2
        )
    
    def generate_remove_dead_code_pattern(self) -> CodePattern:
        """Generate remove dead code refactoring pattern."""
        var_names = random.sample(self.variable_names, k=4)
        
        # Generate original code with dead code
        original_code = f"""def process_{random.choice(['request', 'data', 'input'])}(self, {var_names[0]}):
    {var_names[1]} = {var_names[0]} * 2
    
    # Dead code - never executed
    if False:
        {var_names[2]} = {var_names[1]} + 100
        print("This will never execute")
        return {var_names[2]}
    
    # Unused variable
    {var_names[3]} = "unused"
    
    return {var_names[1]}"""
        
        # Generate refactored code
        refactored_code = f"""def process_{random.choice(['request', 'data', 'input'])}(self, {var_names[0]}):
    {var_names[1]} = {var_names[0]} * 2
    return {var_names[1]}"""
        
        return CodePattern(
            original_code=original_code,
            refactored_code=refactored_code,
            pattern_type="remove_dead_code",
            description="Remove unreachable and unused code",
            quality_improvement={
                "readability": 0.6,
                "maintainability": 0.8,
                "performance": 0.3
            },
            complexity_reduction=0.5
        )
    
    def generate_simplify_conditional_pattern(self) -> CodePattern:
        """Generate simplify conditional refactoring pattern."""
        var_names = random.sample(self.variable_names, k=3)
        
        # Generate original code with complex conditional
        original_code = f"""def check_{random.choice(['status', 'condition', 'state'])}(self, {var_names[0]}, {var_names[1]}):
    if {var_names[0]} > 0:
        if {var_names[1]} is not None:
            if {var_names[1]} != "":
                return True
            else:
                return False
        else:
            return False
    else:
        return False"""
        
        # Generate refactored code
        refactored_code = f"""def check_{random.choice(['status', 'condition', 'state'])}(self, {var_names[0]}, {var_names[1]}):
    return {var_names[0]} > 0 and {var_names[1]} is not None and {var_names[1]} != \"\""""
        
        return CodePattern(
            original_code=original_code,
            refactored_code=refactored_code,
            pattern_type="simplify_conditional",
            description="Simplify nested conditional statements",
            quality_improvement={
                "readability": 0.8,
                "maintainability": 0.7,
                "conciseness": 0.9
            },
            complexity_reduction=0.6
        )
    
    def generate_extract_class_pattern(self) -> CodePattern:
        """Generate extract class refactoring pattern."""
        class_name = random.choice(self.class_names)
        method_names = random.sample(self.method_names, k=3)
        var_names = random.sample(self.variable_names, k=4)
        
        # Generate original code with god class
        original_code = f"""class {class_name}:
    def __init__(self):
        self.{var_names[0]} = []
        self.{var_names[1]} = {{}}
        self.{var_names[2]} = 0
        self.{var_names[3]} = ""
    
    def {method_names[0]}(self, item):
        self.{var_names[0]}.append(item)
        self.{var_names[2]} += 1
        
    def {method_names[1]}(self, key, value):
        self.{var_names[1]}[key] = value
        
    def {method_names[2]}(self, message):
        self.{var_names[3]} = message
        print(f"Log: {{message}}")"""
        
        # Generate refactored code
        extracted_class = f"{random.choice(['Logger', 'Handler', 'Tracker'])}"
        refactored_code = f"""class {extracted_class}:
    def __init__(self):
        self.{var_names[3]} = ""
        
    def {method_names[2]}(self, message):
        self.{var_names[3]} = message
        print(f"Log: {{message}}")

class {class_name}:
    def __init__(self):
        self.{var_names[0]} = []
        self.{var_names[1]} = {{}}
        self.{var_names[2]} = 0
        self.logger = {extracted_class}()
    
    def {method_names[0]}(self, item):
        self.{var_names[0]}.append(item)
        self.{var_names[2]} += 1
        
    def {method_names[1]}(self, key, value):
        self.{var_names[1]}[key] = value"""
        
        return CodePattern(
            original_code=original_code,
            refactored_code=refactored_code,
            pattern_type="extract_class",
            description="Extract related functionality into separate class",
            quality_improvement={
                "separation_of_concerns": 0.8,
                "maintainability": 0.7,
                "reusability": 0.9
            },
            complexity_reduction=0.5
        )
    
    def generate_dataset(self, size: int = 1000) -> List[CodePattern]:
        """Generate synthetic dataset of refactoring patterns."""
        patterns = []
        pattern_generators = [
            self.generate_extract_method_pattern,
            self.generate_extract_variable_pattern,
            self.generate_replace_magic_numbers_pattern,
            self.generate_remove_dead_code_pattern,
            self.generate_simplify_conditional_pattern,
            self.generate_extract_class_pattern
        ]
        
        for i in range(size):
            generator = random.choice(pattern_generators)
            pattern = generator()
            patterns.append(pattern)
            
            if i % 100 == 0:
                print(f"Generated {i} patterns...")
        
        return patterns
    
    def save_dataset(self, patterns: List[CodePattern], output_dir: str):
        """Save dataset to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        json_data = []
        for pattern in patterns:
            json_data.append({
                "original_code": pattern.original_code,
                "refactored_code": pattern.refactored_code,
                "pattern_type": pattern.pattern_type,
                "description": pattern.description,
                "quality_improvement": pattern.quality_improvement,
                "complexity_reduction": pattern.complexity_reduction
            })
        
        with open(os.path.join(output_dir, "refactoring_patterns.json"), "w") as f:
            json.dump(json_data, f, indent=2)
        
        # Save individual files for each pattern type
        pattern_types = {}
        for pattern in patterns:
            if pattern.pattern_type not in pattern_types:
                pattern_types[pattern.pattern_type] = []
            pattern_types[pattern.pattern_type].append(pattern)
        
        for pattern_type, type_patterns in pattern_types.items():
            type_dir = os.path.join(output_dir, pattern_type)
            os.makedirs(type_dir, exist_ok=True)
            
            for i, pattern in enumerate(type_patterns):
                # Save original code
                with open(os.path.join(type_dir, f"original_{i}.py"), "w") as f:
                    f.write(pattern.original_code)
                
                # Save refactored code
                with open(os.path.join(type_dir, f"refactored_{i}.py"), "w") as f:
                    f.write(pattern.refactored_code)
        
        print(f"Dataset saved to {output_dir}")
        print(f"Generated {len(patterns)} patterns across {len(pattern_types)} types")

def main():
    """Main function to generate synthetic dataset."""
    config = {
        "dataset_size": 10000,
        "output_dir": "data/synthetic_dataset"
    }
    
    generator = SyntheticDataGenerator(config)
    patterns = generator.generate_dataset(config["dataset_size"])
    generator.save_dataset(patterns, config["output_dir"])

if __name__ == "__main__":
    main()