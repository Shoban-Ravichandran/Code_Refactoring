# src/models/ensemble_model.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
from .codebert_model import CodeBERTForRefactoring
from .starcoder_model import StarCoderForRefactoring

logger = logging.getLogger(__name__)

class EnsembleRefactoringModel(nn.Module):
    """Ensemble model combining CodeBERT and StarCoder for code refactoring."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Initialize individual models
        self.codebert_model = CodeBERTForRefactoring(config)
        self.starcoder_model = StarCoderForRefactoring(config)
        
        # Ensemble weights
        self.codebert_weight = config["model"]["ensemble"]["codebert_weight"]
        self.starcoder_weight = config["model"]["ensemble"]["starcoder_weight"]
        self.fusion_method = config["model"]["ensemble"]["fusion_method"]
        
        # Learned fusion weights (optional)
        if self.fusion_method == "learned":
            self.fusion_network = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
                nn.Softmax(dim=-1)
            )
        
        # Cross-attention mechanism for feature fusion
        hidden_size = 768  # Assuming similar hidden sizes
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final decision layer
        self.decision_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble model."""
        
        # Get outputs from both models
        codebert_outputs = self.codebert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        starcoder_outputs = self.starcoder_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Extract features for fusion
        codebert_features = codebert_outputs["pooled_output"]
        starcoder_features = starcoder_outputs["hidden_states"][:, -1, :]  # Last token
        
        # Apply cross-attention for feature fusion
        fused_features, attention_weights = self.cross_attention(
            codebert_features.unsqueeze(1),
            starcoder_features.unsqueeze(1),
            starcoder_features.unsqueeze(1)
        )
        
        # Combine features
        combined_features = torch.cat([
            codebert_features,
            fused_features.squeeze(1)
        ], dim=-1)
        
        # Final decision
        ensemble_confidence = self.decision_layer(combined_features)
        
        # Combine losses if available
        ensemble_loss = None
        if codebert_outputs["loss"] is not None and starcoder_outputs["loss"] is not None:
            ensemble_loss = (
                self.codebert_weight * codebert_outputs["loss"] +
                self.starcoder_weight * starcoder_outputs["loss"]
            )
        
        return {
            "loss": ensemble_loss,
            "codebert_outputs": codebert_outputs,
            "starcoder_outputs": starcoder_outputs,
            "ensemble_confidence": ensemble_confidence,
            "attention_weights": attention_weights,
            "combined_features": combined_features
        }
    
    def generate_refactored_code(
        self,
        original_code: str,
        use_voting: bool = True,
        confidence_threshold: float = 0.7,
        **generation_kwargs
    ) -> Dict[str, Union[str, float, List]]:
        """Generate refactored code using ensemble approach."""
        
        # Get predictions from both models
        codebert_result = self.codebert_model.generate_refactored_code(
            original_code, **generation_kwargs
        )
        
        starcoder_result = self.starcoder_model.generate_refactored_code(
            original_code, **generation_kwargs
        )
        
        # Assess code quality for both results
        codebert_quality = self.codebert_model.assess_code_quality(
            codebert_result
        )
        starcoder_quality = self.starcoder_model.evaluate_refactoring_quality(
            original_code, starcoder_result["refactored_code"]
        )
        
        # Calculate ensemble confidence
        ensemble_confidence = self._calculate_ensemble_confidence(
            codebert_result, starcoder_result, 
            codebert_quality, starcoder_quality
        )
        
        # Select best result based on fusion method
        if self.fusion_method == "weighted_average":
            selected_result = self._weighted_selection(
                codebert_result, starcoder_result,
                codebert_quality, starcoder_quality
            )
        elif self.fusion_method == "voting":
            selected_result = self._voting_selection(
                codebert_result, starcoder_result,
                codebert_quality, starcoder_quality
            )
        else:  # confidence_based
            selected_result = self._confidence_based_selection(
                codebert_result, starcoder_result,
                confidence_threshold
            )
        
        return {
            "refactored_code": selected_result["refactored_code"],
            "ensemble_confidence": ensemble_confidence,
            "selected_model": selected_result["model"],
            "codebert_result": codebert_result,
            "starcoder_result": starcoder_result,
            "codebert_quality": codebert_quality,
            "starcoder_quality": starcoder_quality,
            "quality_scores": selected_result["quality_scores"]
        }
    
    def _calculate_ensemble_confidence(
        self,
        codebert_result: str,
        starcoder_result: Dict,
        codebert_quality: Dict,
        starcoder_quality: Dict
    ) -> float:
        """Calculate ensemble confidence score."""
        
        # CodeBERT confidence (based on quality metrics)
        codebert_conf = np.mean([
            codebert_quality.get("readability", 0.5),
            codebert_quality.get("maintainability", 0.5),
            codebert_quality.get("reusability", 0.5)
        ])
        
        # StarCoder confidence
        starcoder_conf = starcoder_result.get("confidence", 0.5)
        
        # Weighted ensemble confidence
        ensemble_conf = (
            self.codebert_weight * codebert_conf +
            self.starcoder_weight * starcoder_conf
        )
        
        return float(ensemble_conf)
    
    def _weighted_selection(
        self,
        codebert_result: str,
        starcoder_result: Dict,
        codebert_quality: Dict,
        starcoder_quality: Dict
    ) -> Dict:
        """Select result based on weighted scores."""
        
        # Calculate weighted scores
        codebert_score = (
            0.3 * codebert_quality.get("readability", 0.5) +
            0.3 * codebert_quality.get("maintainability", 0.5) +
            0.2 * codebert_quality.get("performance", 0.5) +
            0.2 * codebert_quality.get("reusability", 0.5)
        )
        
        starcoder_score = (
            0.3 * starcoder_quality.get("readability", 0.5) +
            0.3 * starcoder_quality.get("maintainability", 0.5) +
            0.2 * starcoder_quality.get("performance", 0.5) +
            0.2 * starcoder_quality.get("correctness", 0.5)
        )
        
        if codebert_score >= starcoder_score:
            return {
                "refactored_code": codebert_result,
                "model": "codebert",
                "quality_scores": codebert_quality,
                "score": codebert_score
            }
        else:
            return {
                "refactored_code": starcoder_result["refactored_code"],
                "model": "starcoder",
                "quality_scores": starcoder_quality,
                "score": starcoder_score
            }
    
    def _voting_selection(
        self,
        codebert_result: str,
        starcoder_result: Dict,
        codebert_quality: Dict,
        starcoder_quality: Dict
    ) -> Dict:
        """Select result based on majority voting of quality metrics."""
        
        codebert_votes = 0
        starcoder_votes = 0
        
        # Vote based on individual metrics
        metrics = ["readability", "maintainability", "performance"]
        
        for metric in metrics:
            codebert_val = codebert_quality.get(metric, 0.5)
            starcoder_val = starcoder_quality.get(metric, 0.5)
            
            if codebert_val > starcoder_val:
                codebert_votes += 1
            else:
                starcoder_votes += 1
        
        if codebert_votes >= starcoder_votes:
            return {
                "refactored_code": codebert_result,
                "model": "codebert",
                "quality_scores": codebert_quality,
                "votes": codebert_votes
            }
        else:
            return {
                "refactored_code": starcoder_result["refactored_code"],
                "model": "starcoder",
                "quality_scores": starcoder_quality,
                "votes": starcoder_votes
            }
    
    def _confidence_based_selection(
        self,
        codebert_result: str,
        starcoder_result: Dict,
        threshold: float
    ) -> Dict:
        """Select result based on confidence threshold."""
        
        starcoder_conf = starcoder_result.get("confidence", 0.5)
        
        if starcoder_conf >= threshold:
            return {
                "refactored_code": starcoder_result["refactored_code"],
                "model": "starcoder",
                "quality_scores": {},
                "confidence": starcoder_conf
            }
        else:
            return {
                "refactored_code": codebert_result,
                "model": "codebert",
                "quality_scores": {},
                "confidence": 1.0 - starcoder_conf
            }
    
    def batch_refactor(
        self,
        code_snippets: List[str],
        **generation_kwargs
    ) -> List[Dict]:
        """Batch refactoring using ensemble approach."""
        
        results = []
        for code in code_snippets:
            result = self.generate_refactored_code(code, **generation_kwargs)
            results.append(result)
            
        return results
    
    def evaluate_ensemble_performance(
        self,
        test_data: List[Dict],
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """Evaluate ensemble performance on test data."""
        
        if metrics is None:
            metrics = ["accuracy", "quality_improvement", "pattern_accuracy"]
        
        results = {
            "ensemble_accuracy": 0.0,
            "codebert_accuracy": 0.0,
            "starcoder_accuracy": 0.0,
            "ensemble_quality": 0.0,
            "pattern_accuracy": 0.0,
            "confidence_correlation": 0.0
        }
        
        ensemble_correct = 0
        codebert_correct = 0
        starcoder_correct = 0
        total_quality = 0
        pattern_correct = 0
        
        for i, sample in enumerate(test_data):
            original_code = sample["original_code"]
            expected_refactored = sample["refactored_code"]
            expected_pattern = sample["pattern_type"]
            
            # Get ensemble prediction
            ensemble_result = self.generate_refactored_code(original_code)
            
            # Evaluate accuracy (simplified - in practice, use AST comparison)
            if self._codes_equivalent(
                ensemble_result["refactored_code"], expected_refactored
            ):
                ensemble_correct += 1
            
            if self._codes_equivalent(
                ensemble_result["codebert_result"], expected_refactored
            ):
                codebert_correct += 1
            
            if self._codes_equivalent(
                ensemble_result["starcoder_result"]["refactored_code"], 
                expected_refactored
            ):
                starcoder_correct += 1
            
            # Evaluate quality improvement
            quality_score = np.mean(list(ensemble_result["quality_scores"].values()))
            total_quality += quality_score
            
            # Evaluate pattern prediction
            predicted_pattern = ensemble_result["starcoder_result"]["predicted_pattern"]
            if predicted_pattern == expected_pattern:
                pattern_correct += 1
        
        # Calculate final metrics
        n_samples = len(test_data)
        results["ensemble_accuracy"] = ensemble_correct / n_samples
        results["codebert_accuracy"] = codebert_correct / n_samples
        results["starcoder_accuracy"] = starcoder_correct / n_samples
        results["ensemble_quality"] = total_quality / n_samples
        results["pattern_accuracy"] = pattern_correct / n_samples
        
        return results
    
    def _codes_equivalent(self, code1: str, code2: str) -> bool:
        """Check if two code snippets are functionally equivalent."""
        # Simplified comparison - in practice, use AST comparison
        # or execute both codes and compare outputs
        
        # Remove whitespace and comments for basic comparison
        import re
        
        def normalize_code(code):
            # Remove comments
            code = re.sub(r'#.*', '', code)
            # Remove extra whitespace
            code = re.sub(r'\s+', ' ', code.strip())
            return code
        
        return normalize_code(code1) == normalize_code(code2)
    
    def save_ensemble(self, save_directory: str):
        """Save the ensemble model."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save individual models
        self.codebert_model.save_model(f"{save_directory}/codebert")
        self.starcoder_model.save_model(f"{save_directory}/starcoder")
        
        # Save ensemble-specific components
        torch.save({
            "fusion_network": self.fusion_network.state_dict() if hasattr(self, 'fusion_network') else None,
            "cross_attention": self.cross_attention.state_dict(),
            "decision_layer": self.decision_layer.state_dict(),
            "config": self.config
        }, f"{save_directory}/ensemble_components.pt")
        
        logger.info(f"Ensemble model saved to {save_directory}")
    
    @classmethod
    def load_ensemble(cls, load_directory: str, config: Dict):
        """Load the ensemble model."""
        model = cls(config)
        
        # Load individual models
        model.codebert_model = CodeBERTForRefactoring.load_model(
            f"{load_directory}/codebert", config
        )
        model.starcoder_model = StarCoderForRefactoring.load_model(
            f"{load_directory}/starcoder", config
        )
        
        # Load ensemble components
        checkpoint = torch.load(f"{load_directory}/ensemble_components.pt")
        
        if checkpoint["fusion_network"] is not None:
            model.fusion_network.load_state_dict(checkpoint["fusion_network"])
        
        model.cross_attention.load_state_dict(checkpoint["cross_attention"])
        model.decision_layer.load_state_dict(checkpoint["decision_layer"])
        
        logger.info(f"Ensemble model loaded from {load_directory}")
        return model