import json
import os
import re
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from codebleu import calc_codebleu
from utils.data_types import TaskResult, TaskType


class NLPProcessor:
    def __init__(self, modality, dataset_dir: str, pred_json_file: str = "prediction.json"):
        self.modality = modality
        self.dataset_dir = dataset_dir + '/nlp'
        self.pred_json_file = pred_json_file
    
    def process(self) -> List[TaskResult]:
        results = []
        
        task_dirs = [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))]
        total_tasks = len(task_dirs)
        processed_tasks = 0
        
        for task_folder in task_dirs:
            folder_path = os.path.join(self.dataset_dir, task_folder)
            annotation_path = os.path.join(folder_path, "annotation.json")
            prediction_path = os.path.join(folder_path, self.pred_json_file)
            
            if not os.path.exists(annotation_path):
                print(f"Skip {task_folder}: annotation.json no exists")
                continue
            
            if not os.path.exists(prediction_path):
                print(f"Skip {task_folder}: {self.pred_json_file} no exists.")
                continue
            
            try:
                with open(annotation_path, "r", encoding="utf-8") as f:
                    task_data = json.load(f)
                
                with open(prediction_path, "r", encoding="utf-8") as f:
                    predictions_data = json.load(f)
                
                task_result = self._evaluate_task(task_data, predictions_data)
                if task_result:
                    results.append(task_result)
                    processed_tasks += 1
                    print(f"Task: {task_folder} (Socre: {task_result.score:.4f})")
                else:
                    print(f"Skip {task_folder}.")
                    
            except Exception as e:
                print(f"Skip {task_folder}: Error - {e}")
                continue
        
        return results
    
    def _evaluate_task(self, task_data: Dict[str, Any], predictions_data: List[Dict]) -> Optional[TaskResult]:
        task_type = task_data.get("type", "")
        task_name = task_data.get("task", "")
        
        pred_map = {pred["id"]: pred for pred in predictions_data}
        
        predictions = []
        references = []
        
        for data_item in task_data["data"]:
            item_id = data_item["id"]
            if item_id not in pred_map:
                continue
            
            pred_item = pred_map[item_id]
            
            if "prediction" in pred_item:
                pred = pred_item["prediction"]
            elif "prediction_final" in pred_item: 
                pred = pred_item["prediction_final"]
            else:
                continue
            
            ref = self._extract_reference(data_item, task_type)
            if ref is None:
                continue
                
            predictions.append(pred)
            references.append(ref)
        
        if not predictions:
            return None
        
        score, metric = self._calculate_metrics(predictions, references, task_type)
        metric = self._convert_metric(metric)
        
        return TaskResult(
            task_name=task_name,
            metric=metric,
            score=score,
            task_type=TaskType.COMPREHENSION
        )
    
    def _extract_reference(self, data_item: Dict[str, Any], task_type: str) -> Any:
        output = data_item.get("output", {})
        
        if task_type == "MultipleChoiceQA":
            return output.get("answer")
        elif task_type == "OpenQA":
            return output.get("answer")
        elif task_type == "Summarization":
            return output.get("summary") or output.get("highlights")
        elif task_type == "Translation":
            if isinstance(output, str):
                return output
            else:
                return output.get("translation")
        elif task_type == "Story Generation":
            return output.get("story")
        elif task_type == "Dialogue":
            return output.get("reference")
        elif task_type == "Code Generation":
            return output.get("response", {}).get("content")
        elif task_type == "Code Repair":
            return output.get("repairCode")
        elif task_type == "Code Defect Detection":
            return str(output.get("target"))
        elif task_type == "Text to SQL":
            return output.get("sql")
        elif task_type == "Code Explanation":
            return output.get("nl")
        elif task_type == "Proof":
            proof_data = output.get("proof", {})
            steps = proof_data.get("steps", [])
            conclusion = proof_data.get("conclusion", "")
            return "\n".join(steps) + f"\nConclusion: {conclusion}"
        elif task_type == "Mathematical Word Problem Solving":
            return output.get("solution", {}).get("final_answer")
        elif task_type == "Paraphrase Generation":
            return output.get("paraphraseSentence")
        elif task_type == "Grammar Correction":
            return output.get("Standard English")
        elif task_type == "Text Style Transfer":
            return output.get("answer")
        elif task_type == "Table-to-Text Generation":
            return output.get("response", {}).get("text")
        elif task_type == "Time Series":
            return output.get("target")
        elif task_type in ["classification", "multiple choice"]:
            return list(output.values())[0].lower() if output else ""
        elif task_type in ["multi label classification", "ner", "extraction", "relation extraction", "event detection", "parsing"]:
            value = list(output.values())[0] if output else ""
            return '<p>'.join(value.lower().split(', ')) if isinstance(value, str) else ""
        else:
            # 默认取第一个值
            return list(output.values())[0] if output else ""
    
    def _calculate_metrics(self, predictions: List, references: List, task_type: str) -> tuple:
        if task_type == "MultipleChoiceQA":
            score = self._exact_match_accuracy(predictions, references)
            return score, "accuracy"
        
        elif task_type == "OpenQA":
            f1_score = self._calculate_f1(predictions, references)
            return f1_score, "f1"
        
        elif task_type == "Summarization":
            rouge_scores = self._rouge_evaluation(predictions, references)
            return rouge_scores["rouge1"], "rouge1"
        
        elif task_type == "Translation":
            rouge_scores = self._rouge_evaluation(predictions, references)
            return rouge_scores["rouge1"], "rouge1"
        
        elif task_type in ["Story Generation", "Dialogue", "Paraphrase Generation", "Grammar Correction", "Text Style Transfer", "Table-to-Text Generation"]:
            bleu_scores = self._bleu_evaluation(predictions, references)
            return bleu_scores["bleu1"], "bleu1"
        
        elif task_type in ["Code Generation", "Code Repair"]:
            try:
                result = calc_codebleu(references, predictions, lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
                return result["codebleu"], "code_bleu"
            except:
                return 0.0, "code_bleu"
        
        elif task_type == "Code Defect Detection":
            score = self._exact_match_accuracy(predictions, references)
            return score, "accuracy"
        
        elif task_type == "Text to SQL":
            score = self._exact_match_accuracy(predictions, references)
            return score, "accuracy"
        
        elif task_type in ["Code Explanation", "Proof"]:
            bleu_scores = self._bleu_evaluation(predictions, references)
            return bleu_scores["bleu1"], "bleu1"
        
        elif task_type == "Mathematical Word Problem Solving":
            score = self._exact_match_accuracy(predictions, references)
            return score, "accuracy"
        
        elif task_type == "Time Series":
            mae = self._mean_absolute_error(predictions, references)
            return mae, "MAE"
        
        elif task_type in ["classification", "multiple choice"]:
            f1_score = self._calculate_micro_f1(predictions, references)
            return f1_score, "micro_f1"
        
        elif task_type in ["multi label classification", "ner", "extraction", "relation extraction", "event detection", "parsing"]:
            f1_score = self._calculate_micro_f1(predictions, references)
            return f1_score, "micro_f1"
        
        else:
            f1_score = self._calculate_f1(predictions, references)
            return f1_score, "f1"
    
    def _exact_match_accuracy(self, predictions: List[str], references: List[str]) -> float:
        correct = 0
        for pred, ref in zip(predictions, references):
            if isinstance(ref, str):
                ref = [ref]
            is_match = False
            for r in ref:
                if str(pred).strip() == str(r).strip():
                    is_match = True
                    break
            if is_match:
                correct += 1
        return correct / len(predictions) if predictions else 0.0
    
    def _calculate_f1(self, predictions: List[str], references: List[str]) -> float:
        def compute_f1(pred: str, ref: str) -> float:
            pred_tokens = str(pred).strip().split()
            ref_tokens = str(ref).strip().split()
            
            common_tokens = set(pred_tokens) & set(ref_tokens)
            num_common = len(common_tokens)
            
            if num_common == 0:
                return 0.0
            
            precision = num_common / len(pred_tokens) if pred_tokens else 0.0
            recall = num_common / len(ref_tokens) if ref_tokens else 0.0
            
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        total_f1 = 0.0
        for pred, ref in zip(predictions, references):
            if isinstance(ref, str):
                ref = [ref]
            max_f1 = 0.0
            for r in ref:
                max_f1 = max(compute_f1(pred, r), max_f1)
            total_f1 += max_f1
        
        return total_f1 / len(predictions) if predictions else 0.0
    
    def _calculate_micro_f1(self, predictions: List[str], references: List[str]) -> float:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(str(pred).strip().split('<p>'))
            ref_tokens = set(str(ref).strip().split("<p>"))
            
            tp = len(pred_tokens & ref_tokens)
            fp = len(pred_tokens - ref_tokens)
            fn = len(ref_tokens - pred_tokens)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        if total_tp == 0:
            return 0.0
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _rouge_evaluation(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_scores, rouge2_scores, rougel_scores = [], [], []
        
        for pred, ref in zip(predictions, references):
            if isinstance(ref, str):
                ref = [ref]
            rouge1, rouge2, rougeL = 0, 0, 0
            for r in ref:
                scores = scorer.score(str(r), str(pred))
                rouge1 = max(scores['rouge1'].fmeasure, rouge1)
                rouge2 = max(scores['rouge2'].fmeasure, rouge2)
                rougeL = max(scores['rougeL'].fmeasure, rougeL)
            rouge1_scores.append(rouge1)
            rouge2_scores.append(rouge2)
            rougel_scores.append(rougeL)
        
        return {
            'rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
            'rouge2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
            'rougeL': sum(rougel_scores) / len(rougel_scores) if rougel_scores else 0.0,
        }
    
    def _bleu_evaluation(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        smoothie = SmoothingFunction().method4
        bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
        
        for pred, ref in zip(predictions, references):
            try:
                hypothesis = nltk.word_tokenize(str(pred))
            except:
                hypothesis = str(pred).split()
                
            if isinstance(ref, str):
                ref = [ref]
                
            bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
            for r in ref:
                try:
                    reference = [nltk.word_tokenize(str(r))]
                except:
                    reference = [str(r).split()]
                    
                try:
                    bleu1 = max(sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothie), bleu1)
                    bleu2 = max(sentence_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie), bleu2)
                    bleu3 = max(sentence_bleu(reference, hypothesis, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothie), bleu3)
                    bleu4 = max(sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie), bleu4)
                except:
                    continue
            
            bleu1_scores.append(bleu1)
            bleu2_scores.append(bleu2)
            bleu3_scores.append(bleu3)
            bleu4_scores.append(bleu4)
        
        return {
            'bleu1': sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0,
            'bleu2': sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0.0,
            'bleu3': sum(bleu3_scores) / len(bleu3_scores) if bleu3_scores else 0.0,
            'bleu4': sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0.0,
        }
    
    def _mean_absolute_error(self, predictions: List[float], references: List[float]) -> float:
        if not predictions:
            return 0.0
        
        error_sum = 0.0
        valid_count = 0
        
        for p, r in zip(predictions, references):
            try:
                error_sum += abs(float(p) - float(r))
                valid_count += 1
            except:
                continue
        
        return error_sum / valid_count if valid_count > 0 else 0.0
    
    def _convert_metric(self, metric: str) -> str:
        m = metric.lower()
        if m == "accuracy":
            return "ACC"
        if m == "f1":
            return "F1"
        if m == "micro_f1":
            return "Micro-F1"
        if m.startswith("rouge"):
            if "l" in m:
                return "ROUGE-L"
            else:
                return "ROUGE-1"
        if m.startswith("bleu"):
            return "BLEU-1"
        if m == "code_bleu":
            return "CodeBLEU"
        return metric.upper()

