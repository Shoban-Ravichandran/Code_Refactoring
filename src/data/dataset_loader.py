# src/data/dataset_loader.py
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import random

logger = logging.getLogger(__name__)

class RefactoringDataset(Dataset):
    """Dataset class for code refactoring training data."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512,
        include_pattern_labels: bool = True
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_pattern_labels = include_pattern_labels
        
        # Pattern type mapping
        self.pattern_to_id = {
            "extract_method": 0,
            "inline_method": 1,
            "extract_variable": 2,
            "inline_variable": 3,
            "move_method": 4,
            "rename_method": 5,
            "extract_class": 6,
            "inline_class": 7,
            "replace_conditional_with_polymorphism": 8,
            "replace_magic_numbers": 9,
            "remove_dead_code": 10,
            "simplify_conditional": 11
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Prepare input text
        original_code = sample["original_code"]
        refactored_code = sample["refactored_code"]
        
        # Create input format: [ORIGINAL] code [REFACTORED] refactored_code
        input_text = f"<ORIGINAL>{original_code}<REFACTORED>{refactored_code}"
        target_text = refactored_code
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        result = {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
            "original_code": original_code,
            "refactored_code": refactored_code
        }
        
        # Add pattern labels if requested
        if self.include_pattern_labels:
            pattern_type = sample.get("pattern_type", "extract_method")
            pattern_id = self.pattern_to_id.get(pattern_type, 0)
            result["pattern_labels"] = torch.tensor(pattern_id, dtype=torch.long)
        
        # Add quality labels if available
        if "quality_improvement" in sample:
            quality_metrics = sample["quality_improvement"]
            quality_tensor = torch.tensor([
                quality_metrics.get("readability", 0.5),
                quality_metrics.get("maintainability", 0.5),
                quality_metrics.get("performance", 0.5),
                quality_metrics.get("reusability", 0.5)
            ], dtype=torch.float)
            result["quality_labels"] = quality_tensor
        
        # Add complexity labels if available
        if "complexity_reduction" in sample:
            result["complexity_labels"] = torch.tensor(
                sample["complexity_reduction"], dtype=torch.float
            )
        
        return result

class RefactoringDatasetLoader:
    """Loads and processes refactoring datasets."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.synthetic_data_config = config.get("synthetic_data", {})
        
    def load_synthetic_dataset(
        self,
        data_path: str = "data/synthetic_dataset/refactoring_patterns.json"
    ) -> Tuple[RefactoringDataset, RefactoringDataset, RefactoringDataset]:
        """Load synthetic dataset and split into train/val/test."""
        
        logger.info(f"Loading synthetic dataset from {data_path}")
        
        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Shuffle data
        random.shuffle(data)
        
        # Split data
        train_split = self.synthetic_data_config.get("train_split", 0.8)
        val_split = self.synthetic_data_config.get("val_split", 0.1)
        
        total_size = len(data)
        train_size = int(total_size * train_split)
        val_size = int(total_size * val_split)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        logger.info(f"Dataset split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Initialize tokenizers
        codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        starcoder_tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
        
        # Add padding token if needed
        if codebert_tokenizer.pad_token is None:
            codebert_tokenizer.pad_token = codebert_tokenizer.eos_token
        if starcoder_tokenizer.pad_token is None:
            starcoder_tokenizer.pad_token = starcoder_tokenizer.eos_token
        
        # Create datasets (using CodeBERT tokenizer as default)
        max_length = self.config.get("model", {}).get("codebert", {}).get("max_length", 512)
        
        train_dataset = RefactoringDataset(
            train_data, codebert_tokenizer, max_length=max_length
        )
        val_dataset = RefactoringDataset(
            val_data, codebert_tokenizer, max_length=max_length
        )
        test_dataset = RefactoringDataset(
            test_data, codebert_tokenizer, max_length=max_length
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def load_real_world_dataset(
        self,
        data_path: str = "data/real_world_samples"
    ) -> RefactoringDataset:
        """Load real-world refactoring examples."""
        
        logger.info(f"Loading real-world dataset from {data_path}")
        
        data = []
        data_dir = Path(data_path)
        
        # Look for JSON files containing refactoring pairs
        for json_file in data_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    data.extend(file_data)
                else:
                    data.append(file_data)
        
        # Look for paired Python files (original and refactored)
        for original_file in data_dir.glob("*_original.py"):
            refactored_file = original_file.with_name(
                original_file.name.replace("_original.py", "_refactored.py")
            )
            
            if refactored_file.exists():
                with open(original_file, 'r') as f:
                    original_code = f.read()
                with open(refactored_file, 'r') as f:
                    refactored_code = f.read()
                
                data.append({
                    "original_code": original_code,
                    "refactored_code": refactored_code,
                    "pattern_type": "mixed",
                    "description": f"Real-world refactoring from {original_file.name}"
                })
        
        logger.info(f"Loaded {len(data)} real-world samples")
        
        # Create dataset
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        max_length = self.config.get("model", {}).get("codebert", {}).get("max_length", 512)
        
        return RefactoringDataset(data, tokenizer, max_length=max_length)
    
    def create_data_loaders(
        self,
        train_dataset: RefactoringDataset,
        val_dataset: RefactoringDataset,
        test_dataset: RefactoringDataset,
        batch_size: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for training."""
        
        if batch_size is None:
            batch_size = self.config.get("training", {}).get("batch_size", 16)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def create_augmented_dataset(
        self,
        base_dataset: RefactoringDataset,
        augmentation_factor: int = 2
    ) -> RefactoringDataset:
        """Create augmented dataset with additional variations."""
        
        logger.info(f"Creating augmented dataset with factor {augmentation_factor}")
        
        augmented_data = []
        
        for sample in base_dataset.data:
            # Add original sample
            augmented_data.append(sample)
            
            # Create augmented variations
            for _ in range(augmentation_factor - 1):
                augmented_sample = self._augment_sample(sample)
                augmented_data.append(augmented_sample)
        
        logger.info(f"Augmented dataset size: {len(augmented_data)}")
        
        # Create new dataset with augmented data
        return RefactoringDataset(
            augmented_data,
            base_dataset.tokenizer,
            max_length=base_dataset.max_length,
            include_pattern_labels=base_dataset.include_pattern_labels
        )
    
    def _augment_sample(self, sample: Dict) -> Dict:
        """Apply augmentation to a single sample."""
        augmented = sample.copy()
        
        # Variable name variations
        original_code = sample["original_code"]
        refactored_code = sample["refactored_code"]
        
        # Simple variable name substitution
        variable_mappings = {
            "data": "info",
            "result": "output",
            "value": "val",
            "item": "element",
            "temp": "tmp",
            "count": "counter",
            "index": "idx",
            "length": "size"
        }
        
        for old_var, new_var in variable_mappings.items():
            if old_var in original_code:
                original_code = original_code.replace(old_var, new_var)
                refactored_code = refactored_code.replace(old_var, new_var)
                break
        
        augmented["original_code"] = original_code
        augmented["refactored_code"] = refactored_code
        
        # Slightly vary quality metrics
        if "quality_improvement" in augmented:
            quality = augmented["quality_improvement"]
            for metric in quality:
                # Add small random variation
                variation = random.uniform(-0.1, 0.1)
                quality[metric] = max(0.0, min(1.0, quality[metric] + variation))
        
        return augmented
    
    def load_benchmark_dataset(self, benchmark_name: str = "refactoring_miner") -> RefactoringDataset:
        """Load benchmark dataset for evaluation."""
        
        benchmark_path = f"data/benchmarks/{benchmark_name}"
        logger.info(f"Loading benchmark dataset: {benchmark_name}")
        
        data = []
        
        # Load benchmark-specific format
        if benchmark_name == "refactoring_miner":
            # RefactoringMiner format
            data = self._load_refactoring_miner_format(benchmark_path)
        elif benchmark_name == "defects4j":
            # Defects4J format
            data = self._load_defects4j_format(benchmark_path)
        else:
            # Generic format
            data = self._load_generic_benchmark(benchmark_path)
        
        logger.info(f"Loaded {len(data)} benchmark samples")
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        max_length = self.config.get("model", {}).get("codebert", {}).get("max_length", 512)
        
        return RefactoringDataset(data, tokenizer, max_length=max_length)
    
    def _load_refactoring_miner_format(self, data_path: str) -> List[Dict]:
        """Load RefactoringMiner benchmark format."""
        data = []
        
        benchmark_file = os.path.join(data_path, "refactorings.json")
        if os.path.exists(benchmark_file):
            with open(benchmark_file, 'r') as f:
                refactorings = json.load(f)
            
            for refactoring in refactorings:
                if "before" in refactoring and "after" in refactoring:
                    data.append({
                        "original_code": refactoring["before"],
                        "refactored_code": refactoring["after"],
                        "pattern_type": refactoring.get("type", "unknown"),
                        "description": refactoring.get("description", ""),
                        "source": "RefactoringMiner"
                    })
        
        return data
    
    def _load_defects4j_format(self, data_path: str) -> List[Dict]:
        """Load Defects4J benchmark format."""
        data = []
        
        # Defects4J typically has paired buggy/fixed versions
        for bug_dir in Path(data_path).glob("*"):
            if bug_dir.is_dir():
                buggy_file = bug_dir / "buggy.java"
                fixed_file = bug_dir / "fixed.java"
                
                if buggy_file.exists() and fixed_file.exists():
                    with open(buggy_file, 'r') as f:
                        buggy_code = f.read()
                    with open(fixed_file, 'r') as f:
                        fixed_code = f.read()
                    
                    data.append({
                        "original_code": buggy_code,
                        "refactored_code": fixed_code,
                        "pattern_type": "bug_fix",
                        "description": f"Defects4J bug fix: {bug_dir.name}",
                        "source": "Defects4J"
                    })
        
        return data
    
    def _load_generic_benchmark(self, data_path: str) -> List[Dict]:
        """Load generic benchmark format."""
        data = []
        
        # Look for JSON files
        for json_file in Path(data_path).glob("*.json"):
            with open(json_file, 'r') as f:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    data.extend(file_data)
                else:
                    data.append(file_data)
        
        return data
    
    def create_cross_validation_splits(
        self,
        dataset: RefactoringDataset,
        n_folds: int = 5
    ) -> List[Tuple[RefactoringDataset, RefactoringDataset]]:
        """Create cross-validation splits."""
        
        logger.info(f"Creating {n_folds}-fold cross-validation splits")
        
        data = dataset.data.copy()
        random.shuffle(data)
        
        fold_size = len(data) // n_folds
        splits = []
        
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_folds - 1 else len(data)
            
            # Validation set
            val_data = data[start_idx:end_idx]
            
            # Training set (everything else)
            train_data = data[:start_idx] + data[end_idx:]
            
            train_dataset = RefactoringDataset(
                train_data,
                dataset.tokenizer,
                max_length=dataset.max_length,
                include_pattern_labels=dataset.include_pattern_labels
            )
            
            val_dataset = RefactoringDataset(
                val_data,
                dataset.tokenizer,
                max_length=dataset.max_length,
                include_pattern_labels=dataset.include_pattern_labels
            )
            
            splits.append((train_dataset, val_dataset))
        
        return splits
    
    def create_stratified_split(
        self,
        dataset: RefactoringDataset,
        test_size: float = 0.2
    ) -> Tuple[RefactoringDataset, RefactoringDataset]:
        """Create stratified split based on pattern types."""
        
        from collections import defaultdict
        
        # Group data by pattern type
        pattern_groups = defaultdict(list)
        for sample in dataset.data:
            pattern_type = sample.get("pattern_type", "unknown")
            pattern_groups[pattern_type].append(sample)
        
        train_data = []
        test_data = []
        
        # Split each pattern group
        for pattern_type, samples in pattern_groups.items():
            random.shuffle(samples)
            split_idx = int(len(samples) * (1 - test_size))
            
            train_data.extend(samples[:split_idx])
            test_data.extend(samples[split_idx:])
        
        # Shuffle the final datasets
        random.shuffle(train_data)
        random.shuffle(test_data)
        
        train_dataset = RefactoringDataset(
            train_data,
            dataset.tokenizer,
            max_length=dataset.max_length,
            include_pattern_labels=dataset.include_pattern_labels
        )
        
        test_dataset = RefactoringDataset(
            test_data,
            dataset.tokenizer,
            max_length=dataset.max_length,
            include_pattern_labels=dataset.include_pattern_labels
        )
        
        logger.info(f"Stratified split - Train: {len(train_data)}, Test: {len(test_data)}")
        
        return train_dataset, test_dataset
    
    def get_dataset_statistics(self, dataset: RefactoringDataset) -> Dict:
        """Get statistics about the dataset."""
        
        stats = {
            "total_samples": len(dataset),
            "pattern_distribution": defaultdict(int),
            "code_length_stats": {
                "original": {"min": float('inf'), "max": 0, "avg": 0},
                "refactored": {"min": float('inf'), "max": 0, "avg": 0}
            },
            "quality_metrics": {
                "readability": [],
                "maintainability": [],
                "performance": [],
                "reusability": []
            }
        }
        
        total_original_length = 0
        total_refactored_length = 0
        
        for sample in dataset.data:
            # Pattern distribution
            pattern_type = sample.get("pattern_type", "unknown")
            stats["pattern_distribution"][pattern_type] += 1
            
            # Code length statistics
            original_length = len(sample["original_code"])
            refactored_length = len(sample["refactored_code"])
            
            total_original_length += original_length
            total_refactored_length += refactored_length
            
            # Update min/max
            stats["code_length_stats"]["original"]["min"] = min(
                stats["code_length_stats"]["original"]["min"], original_length
            )
            stats["code_length_stats"]["original"]["max"] = max(
                stats["code_length_stats"]["original"]["max"], original_length
            )
            stats["code_length_stats"]["refactored"]["min"] = min(
                stats["code_length_stats"]["refactored"]["min"], refactored_length
            )
            stats["code_length_stats"]["refactored"]["max"] = max(
                stats["code_length_stats"]["refactored"]["max"], refactored_length
            )
            
            # Quality metrics
            if "quality_improvement" in sample:
                quality = sample["quality_improvement"]
                for metric in ["readability", "maintainability", "performance", "reusability"]:
                    if metric in quality:
                        stats["quality_metrics"][metric].append(quality[metric])
        
        # Calculate averages
        if len(dataset) > 0:
            stats["code_length_stats"]["original"]["avg"] = total_original_length / len(dataset)
            stats["code_length_stats"]["refactored"]["avg"] = total_refactored_length / len(dataset)
            
            for metric in stats["quality_metrics"]:
                if stats["quality_metrics"][metric]:
                    values = stats["quality_metrics"][metric]
                    stats["quality_metrics"][metric] = {
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "count": len(values)
                    }
        
        # Convert defaultdict to regular dict
        stats["pattern_distribution"] = dict(stats["pattern_distribution"])
        
        return stats