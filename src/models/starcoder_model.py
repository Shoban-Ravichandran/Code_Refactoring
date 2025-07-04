# src/models/starcoder_model.py
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    TrainingArguments, Trainer, GenerationConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class StarCoderForRefactoring(nn.Module):
    """StarCoder model fine-tuned for code refactoring tasks."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.model_name = config["model"]["starcoder"]["model_name"]
        self.max_length = config["model"]["starcoder"]["max_length"]
        
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # Add special tokens for refactoring
        special_tokens = {
            "additional_special_tokens": [
                "<REFACTOR_START>", "<REFACTOR_END>", "<ORIGINAL_CODE>", 
                "<REFACTORED_CODE>", "<PATTERN_TYPE>", "<QUALITY_METRICS>",
                "<EXTRACT_METHOD>", "<INLINE_METHOD>", "<EXTRACT_VARIABLE>",
                "<REMOVE_DEAD_CODE>", "<SIMPLIFY_CONDITIONAL>", "<EXTRACT_CLASS>"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        
        # Apply LoRA for efficient fine-tuning
        self.setup_lora()
        
        # Additional task-specific heads
        hidden_size = self.base_model.config.hidden_size
        self.refactoring_confidence_head = nn.Linear(hidden_size, 1)
        self.pattern_type_head = nn.Linear(hidden_size, 12)  # 12 refactoring patterns
        
        self.dropout = nn.Dropout(config["model"]["starcoder"]["dropout"])
        
    def setup_lora(self):
        """Setup LoRA configuration for efficient fine-tuning."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config["model"]["lora"]["r"],
            lora_alpha=self.config["model"]["lora"]["lora_alpha"],
            lora_dropout=self.config["model"]["lora"]["lora_dropout"],
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        self.base_model = get_peft_model(self.base_model, lora_config)
        logger.info("LoRA configuration applied to StarCoder model")
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pattern_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )
        
        # Get last hidden states for additional tasks
        hidden_states = outputs.hidden_states[-1]
        
        # Pool hidden states (use last token representation)
        pooled_output = hidden_states[:, -1, :]
        pooled_output = self.dropout(pooled_output)
        
        # Additional task predictions
        confidence_score = torch.sigmoid(self.refactoring_confidence_head(pooled_output))
        pattern_logits = self.pattern_type_head(pooled_output)
        
        # Calculate additional losses if labels provided
        additional_loss = 0
        if pattern_labels is not None:
            pattern_loss_fct = nn.CrossEntropyLoss()
            pattern_loss = pattern_loss_fct(pattern_logits, pattern_labels)
            additional_loss += pattern_loss
            
        # Combine losses
        total_loss = outputs.loss
        if additional_loss > 0:
            total_loss = outputs.loss + 0.1 * additional_loss
        
        return {
            "loss": total_loss,
            "logits": outputs.logits,
            "hidden_states": hidden_states,
            "confidence_score": confidence_score,
            "pattern_logits": pattern_logits,
            "past_key_values": outputs.past_key_values if hasattr(outputs, 'past_key_values') else None
        }
    
    def generate_refactored_code(
        self,
        original_code: str,
        pattern_type: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 4,
        do_sample: bool = True
    ) -> Dict[str, Union[str, float]]:
        """Generate refactored code with confidence score."""
        
        # Create prompt with pattern information
        if pattern_type:
            prompt = f"<REFACTOR_START><PATTERN_TYPE>{pattern_type}</PATTERN_TYPE>\n<ORIGINAL_CODE>\n{original_code}\n</ORIGINAL_CODE>\n<REFACTORED_CODE>\n"
        else:
            prompt = f"<REFACTOR_START><ORIGINAL_CODE>\n{original_code}\n</ORIGINAL_CODE>\n<REFACTORED_CODE>\n"
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length - max_new_tokens,
            truncation=True,
            padding=False
        )
        
        with torch.no_grad():
            # Generate refactored code
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
            
            generated_ids = self.base_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config
            )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )
            
            # Extract refactored code
            if "<REFACTORED_CODE>" in generated_text:
                refactored_part = generated_text.split("<REFACTORED_CODE>")[-1]
                if "<REFACTOR_END>" in refactored_part:
                    refactored_code = refactored_part.split("<REFACTOR_END>")[0].strip()
                else:
                    refactored_code = refactored_part.strip()
            else:
                # Fallback: use everything after the original prompt
                refactored_code = generated_text[len(prompt):].strip()
            
            # Get confidence score
            outputs = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            confidence = outputs["confidence_score"].item()
            
            # Predict pattern type if not provided
            if not pattern_type:
                pattern_probs = torch.softmax(outputs["pattern_logits"], dim=-1)
                pattern_idx = torch.argmax(pattern_probs, dim=-1).item()
                pattern_names = [
                    "extract_method", "inline_method", "extract_variable", "inline_variable",
                    "move_method", "rename_method", "extract_class", "inline_class",
                    "replace_conditional_with_polymorphism", "replace_magic_numbers",
                    "remove_dead_code", "simplify_conditional"
                ]
                predicted_pattern = pattern_names[pattern_idx]
            else:
                predicted_pattern = pattern_type
        
        return {
            "refactored_code": refactored_code,
            "confidence": confidence,
            "predicted_pattern": predicted_pattern,
            "original_prompt": prompt
        }
    
    def batch_refactor(
        self,
        code_snippets: List[str],
        pattern_types: Optional[List[str]] = None,
        **generation_kwargs
    ) -> List[Dict[str, Union[str, float]]]:
        """Batch refactoring for multiple code snippets."""
        
        results = []
        for i, code in enumerate(code_snippets):
            pattern_type = pattern_types[i] if pattern_types else None
            result = self.generate_refactored_code(
                code, pattern_type, **generation_kwargs
            )
            results.append(result)
            
        return results
    
    def evaluate_refactoring_quality(
        self,
        original_code: str,
        refactored_code: str
    ) -> Dict[str, float]:
        """Evaluate the quality of refactoring."""
        
        # Create evaluation prompt
        prompt = f"""<REFACTOR_START>
<ORIGINAL_CODE>
{original_code}
</ORIGINAL_CODE>
<REFACTORED_CODE>
{refactored_code}
</REFACTORED_CODE>
Evaluate the refactoring quality on a scale of 0-1 for:
1. Readability improvement:
2. Maintainability improvement:
3. Performance impact:
4. Code correctness:
"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.base_model.generate(
                inputs["input_ids"],
                max_new_tokens=200,
                temperature=0.3,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            evaluation_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse evaluation scores (simplified parsing)
            scores = {
                "readability": 0.5,
                "maintainability": 0.5,
                "performance": 0.5,
                "correctness": 0.5
            }
            
            # Basic parsing of generated evaluation
            lines = evaluation_text.split('\n')
            for line in lines:
                if "Readability improvement:" in line:
                    try:
                        scores["readability"] = float(line.split(':')[-1].strip())
                    except:
                        pass
                elif "Maintainability improvement:" in line:
                    try:
                        scores["maintainability"] = float(line.split(':')[-1].strip())
                    except:
                        pass
                elif "Performance impact:" in line:
                    try:
                        scores["performance"] = float(line.split(':')[-1].strip())
                    except:
                        pass
                elif "Code correctness:" in line:
                    try:
                        scores["correctness"] = float(line.split(':')[-1].strip())
                    except:
                        pass
        
        return scores
    
    def predict_refactoring_pattern(self, code: str) -> Tuple[str, float]:
        """Predict the best refactoring pattern for given code."""
        
        prompt = f"<REFACTOR_START><ORIGINAL_CODE>\n{code}\n</ORIGINAL_CODE>\nBest refactoring pattern:"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            
            pattern_probs = torch.softmax(outputs["pattern_logits"], dim=-1)
            pattern_idx = torch.argmax(pattern_probs, dim=-1).item()
            confidence = pattern_probs[0, pattern_idx].item()
            
            pattern_names = [
                "extract_method", "inline_method", "extract_variable", "inline_variable",
                "move_method", "rename_method", "extract_class", "inline_class",
                "replace_conditional_with_polymorphism", "replace_magic_numbers",
                "remove_dead_code", "simplify_conditional"
            ]
            
            return pattern_names[pattern_idx], confidence
    
    def generate_explanation(self, original_code: str, refactored_code: str) -> str:
        """Generate explanation for the refactoring."""
        
        prompt = f"""<REFACTOR_START>
<ORIGINAL_CODE>
{original_code}
</ORIGINAL_CODE>
<REFACTORED_CODE>
{refactored_code}
</REFACTORED_CODE>
Explain the refactoring changes and benefits:
"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length - 300,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.base_model.generate(
                inputs["input_ids"],
                max_new_tokens=300,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract explanation part
            if "Explain the refactoring changes and benefits:" in explanation:
                explanation = explanation.split("Explain the refactoring changes and benefits:")[-1].strip()
            
            return explanation
    
    def assess_code_complexity(self, code: str) -> Dict[str, float]:
        """Assess code complexity using the model."""
        
        prompt = f"""<REFACTOR_START>
<ORIGINAL_CODE>
{code}
</ORIGINAL_CODE>
Assess complexity metrics (0-1 scale):
1. Cyclomatic complexity:
2. Readability:
3. Maintainability:
4. Testability:
"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.base_model.generate(
                inputs["input_ids"],
                max_new_tokens=150,
                temperature=0.3,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            assessment_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse assessment scores
            scores = {
                "cyclomatic_complexity": 0.5,
                "readability": 0.5,
                "maintainability": 0.5,
                "testability": 0.5
            }
            
            lines = assessment_text.split('\n')
            for line in lines:
                if "Cyclomatic complexity:" in line:
                    try:
                        scores["cyclomatic_complexity"] = float(line.split(':')[-1].strip())
                    except:
                        pass
                elif "Readability:" in line:
                    try:
                        scores["readability"] = float(line.split(':')[-1].strip())
                    except:
                        pass
                elif "Maintainability:" in line:
                    try:
                        scores["maintainability"] = float(line.split(':')[-1].strip())
                    except:
                        pass
                elif "Testability:" in line:
                    try:
                        scores["testability"] = float(line.split(':')[-1].strip())
                    except:
                        pass
        
        return scores
    
    def suggest_refactoring_steps(self, code: str, target_pattern: str) -> List[str]:
        """Suggest step-by-step refactoring process."""
        
        prompt = f"""<REFACTOR_START>
<ORIGINAL_CODE>
{code}
</ORIGINAL_CODE>
<PATTERN_TYPE>{target_pattern}</PATTERN_TYPE>
Provide step-by-step refactoring instructions:
1.
"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length - 200,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.base_model.generate(
                inputs["input_ids"],
                max_new_tokens=200,
                temperature=0.5,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            steps_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract steps
            if "Provide step-by-step refactoring instructions:" in steps_text:
                steps_part = steps_text.split("Provide step-by-step refactoring instructions:")[-1]
                
                # Parse numbered steps
                steps = []
                lines = steps_part.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('1.')):
                        steps.append(line)
                
                return steps
        
        return ["No specific steps generated"]
    
    def save_model(self, save_directory: str):
        """Save the fine-tuned model."""
        self.base_model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        # Save additional components
        torch.save({
            "refactoring_confidence_head": self.refactoring_confidence_head.state_dict(),
            "pattern_type_head": self.pattern_type_head.state_dict(),
            "config": self.config
        }, f"{save_directory}/starcoder_additional_components.pt")
        
        logger.info(f"StarCoder model saved to {save_directory}")
    
    @classmethod
    def load_model(cls, load_directory: str, config: Dict):
        """Load a fine-tuned model."""
        model = cls(config)
        
        # Load base model and tokenizer
        model.base_model = AutoModelForCausalLM.from_pretrained(
            load_directory,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.tokenizer = AutoTokenizer.from_pretrained(load_directory, trust_remote_code=True)
        
        # Load additional components
        checkpoint = torch.load(f"{load_directory}/starcoder_additional_components.pt")
        model.refactoring_confidence_head.load_state_dict(
            checkpoint["refactoring_confidence_head"]
        )
        model.pattern_type_head.load_state_dict(checkpoint["pattern_type_head"])
        
        logger.info(f"StarCoder model loaded from {load_directory}")
        return model


class StarCoderTrainer:
    """Trainer class for StarCoder model."""
    
    def __init__(self, model: StarCoderForRefactoring, config: Dict):
        self.model = model
        self.config = config
        
    def create_training_args(self) -> TrainingArguments:
        """Create training arguments."""
        return TrainingArguments(
            output_dir="./models/starcoder_checkpoints",
            per_device_train_batch_size=self.config["training"]["batch_size"] // 2,  # Larger model
            per_device_eval_batch_size=self.config["training"]["batch_size"] // 2,
            learning_rate=self.config["training"]["learning_rate"] * 0.5,  # Lower LR for larger model
            num_train_epochs=self.config["training"]["num_epochs"],
            warmup_steps=self.config["training"]["warmup_steps"],
            weight_decay=self.config["training"]["weight_decay"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"] * 2,
            max_grad_norm=self.config["training"]["max_grad_norm"],
            fp16=True,  # Enable for memory efficiency
            gradient_checkpointing=True,
            evaluation_strategy="steps",
            eval_steps=self.config["validation"]["eval_steps"],
            save_steps=self.config["validation"]["save_steps"],
            logging_steps=self.config["validation"]["logging_steps"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,  # Save fewer checkpoints due to model size
            report_to="wandb",
            run_name="starcoder-refactoring",
            dataloader_pin_memory=False,  # Reduce memory usage
            remove_unused_columns=False
        )
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        
        # Calculate perplexity
        shift_logits = predictions[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                       shift_labels.view(-1))
        
        # Mask out padding tokens
        mask = (shift_labels != -100).view(-1)
        loss = loss[mask]
        
        perplexity = torch.exp(loss.mean()).item()
        
        return {
            "perplexity": perplexity,
            "eval_loss": loss.mean().item()
        }
    
    def prepare_dataset_for_training(self, dataset):
        """Prepare dataset specifically for StarCoder training."""
        
        def tokenize_function(examples):
            # Create input-output pairs for causal language modeling
            inputs = []
            
            for i in range(len(examples["original_code"])):
                original = examples["original_code"][i]
                refactored = examples["refactored_code"][i]
                pattern = examples.get("pattern_type", ["extract_method"])[i]
                
                # Format as training sequence
                input_text = f"<REFACTOR_START><PATTERN_TYPE>{pattern}</PATTERN_TYPE>\n<ORIGINAL_CODE>\n{original}\n</ORIGINAL_CODE>\n<REFACTORED_CODE>\n{refactored}<REFACTOR_END>"
                inputs.append(input_text)
            
            # Tokenize
            model_inputs = self.model.tokenizer(
                inputs,
                max_length=self.model.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Labels are the same as input_ids for causal LM
            model_inputs["labels"] = model_inputs["input_ids"].clone()
            
            return model_inputs
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def fine_tune_with_lora(self, train_dataset, eval_dataset):
        """Fine-tune the model using LoRA."""
        
        # Prepare datasets
        train_dataset = self.prepare_dataset_for_training(train_dataset)
        eval_dataset = self.prepare_dataset_for_training(eval_dataset)
        
        # Create trainer
        from transformers import Trainer, DataCollatorForLanguageModeling
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model.tokenizer,
            mlm=False  # We're doing causal language modeling
        )
        
        training_args = self.create_training_args()
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Start training
        trainer.train()
        
        return trainer
    
    def evaluate_model_performance(self, test_dataset):
        """Evaluate model performance on test data."""
        
        metrics = {
            "generation_quality": 0.0,
            "pattern_accuracy": 0.0,
            "confidence_correlation": 0.0,
            "bleu_score": 0.0
        }
        
        total_samples = len(test_dataset)
        correct_patterns = 0
        total_bleu = 0.0
        
        for sample in test_dataset:
            original_code = sample["original_code"]
            expected_refactored = sample["refactored_code"]
            expected_pattern = sample["pattern_type"]
            
            # Generate refactoring
            result = self.model.generate_refactored_code(original_code)
            
            # Check pattern prediction
            predicted_pattern, confidence = self.model.predict_refactoring_pattern(original_code)
            if predicted_pattern == expected_pattern:
                correct_patterns += 1
            
            # Calculate BLEU score (simplified)
            try:
                from nltk.translate.bleu_score import sentence_bleu
                reference = [expected_refactored.split()]
                candidate = result["refactored_code"].split()
                bleu = sentence_bleu(reference, candidate)
                total_bleu += bleu
            except ImportError:
                # Fallback similarity metric
                similarity = self._calculate_similarity(expected_refactored, result["refactored_code"])
                total_bleu += similarity
        
        metrics["pattern_accuracy"] = correct_patterns / total_samples
        metrics["bleu_score"] = total_bleu / total_samples
        metrics["generation_quality"] = total_bleu / total_samples  # Use BLEU as proxy
        
        return metrics
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0