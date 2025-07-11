import json
import os
import tqdm
from typing import List, Dict, Any
import nltk
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dataclasses import dataclass
from abc import ABC, abstractmethod
from transformers import pipeline
from rouge_score import rouge_scorer
from codebleu import calc_codebleu
import math
import numpy as np
import jieba

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


class LLMModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_time_series = False
        self.timesfm_model = None  # timesfm时序模型

        if "timesfm" in model_name.lower():
            import timesfm
            self.is_time_series = True
            self.tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu",
                    per_core_batch_size=32,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=model_name),
            )

        elif "qwen" in model_name.lower() or "gemma" in model_name.lower() or "internlm" in model_name.lower() or "vicuna" in model_name.lower() or "gpt" in model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
            self.copied_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
            self.model = self.model.eval()

        elif "chatglm" in model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
            self.model = self.model.eval()

        else:
            self.pipeline = pipeline("text-generation", model=model_name, device_map="auto", trust_remote_code=True)
        
    
    def generate(self, prompt: str, max_new_tokens=256) -> str:
        if self.is_time_series:
            raise NotImplementedError("This model is a time-series model. Please call generate_for_timeseries() instead of generate().")
        
        if "vicuna" in self.model_name.lower() or "gpt" in self.model_name.lower():
            inputs = self.tokenizer(prompt, return_tensors="pt")
            generate_ids = self.model.generate(inputs.input_ids.cuda(), max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return output

        elif "llama" in self.model_name.lower():
            self.messages = [
                {"role": "system", "content": "You are a helpful and useful AI assistant."}, 
                {"role": "user", "content":prompt }
            ]
            prompt = self.pipeline.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
            terminators = [
                self.pipeline.tokenizer.eos_token_id,
                self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            output = self.pipeline(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1, 
                                pad_token_id = self.pipeline.tokenizer.eos_token_id, 
                                return_full_text=False, eos_token_id=terminators)
            return output[0]["generated_text"]
            
        elif "qwen" in self.model_name.lower():
            self.messages = [
                {"role": "system", "content": "You are a helpful and useful AI assistant."}, 
                {"role": "user", "content": prompt}
            ]
            prompt = self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
            generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        
        elif "gemma" in self.model_name.lower():
            self.messages = [
                {"role": "user", "content": prompt}
            ]
            prompt = self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
            generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        
        elif "chatglm" in self.model_name.lower() or "internlm" in self.model_name.lower():
            response, _ = self.model.chat(self.tokenizer, prompt, history=[])
            return response
            
    def generate_for_timeseries(
        self, 
        series_data: List[float], 
        horizon: int = 1, 
        freq: int = 0
    ) -> List[float]:
        if self.is_time_series and self.tfm is not None:
            forecast_input = [series_data]
            frequency_input = [freq]

            point_forecast, _ = self.tfm.forecast(
                forecast_input,
                freq=frequency_input
            )

            forecast_result = point_forecast[0]
            if horizon < len(forecast_result):
                forecast_result = forecast_result[:horizon]
            return forecast_result.tolist()
        
        else:
            prompt = (
                "You are a time-series forecasting assistant.\n"
                f"The historical data points are: {series_data}.\n"
                f"Please predict the next {horizon} future data point(s) directly without other words based on the historical trend.\n\n"
                "Format your answer as a list of floats, e.g. `[3.1415, 2.7182]`.\n"
                "Answer:"
            )
            
            raw_response = self.generate(prompt, max_new_tokens=64)
            import re
            pattern = r"\[([\d\.\,\s\-eE]+)\]"
            match = re.search(pattern, raw_response)
            if not match:
                print("Warning: LLM output not in expected format, fallback to 0.0")
                return [0.0] * horizon
            
            numbers_str = match.group(1)
            raw_nums = re.split(r"[\s,]+", numbers_str.strip())
            parsed_vals = []
            for val in raw_nums:
                try:
                    parsed_vals.append(float(val))
                except ValueError:
                    continue
            
            # 如果预测数量不够 horizon，就做填充或截断
            if len(parsed_vals) < horizon:
                # 填充
                while len(parsed_vals) < horizon:
                    parsed_vals.append(parsed_vals[-1] if parsed_vals else 0.0)
            elif len(parsed_vals) > horizon:
                parsed_vals = parsed_vals[:horizon]
            
            return parsed_vals
        

@dataclass
class Instance:
    input: Dict[str, Any]
    output: Dict[str, Any]
    id: str

class BaseTask(ABC):
    def __init__(self, task_data: Dict[str, Any], model: LLMModel):
        self.task_data = task_data
        self.model = model
        self.data = self._parse_data(task_data)
    
    @abstractmethod
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        pass
    
    @abstractmethod
    def run_inference(self):
        pass


class MultipleChoiceQA(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output={}, id=d["id"]) 
                for d in task_data["data"]]
    
    def run_inference(self):
        self.predictions = []
        for inst in tqdm.tqdm(self.data):
            question = inst.input["question"]
            options = inst.input["options"]
            options_chars = [chr(65 + i) for i in range(len(options))]
            prompt = f"Question: {question}\nOptions:\n"
            for i, opt in enumerate(options):
                prompt += options_chars[i] + ". " + opt + "\n"
                
            if self.task_data["task"] == "Causal Reasoning":
                prompt += f"{question}\nPlease substitute yourself into the above scenario and select the most likely cause and effect outcome. "
            prompt += r'Please answer the question and output it strictly in the following format: "The final answer is $\boxed{your choice}$" at the end of the sentence.'
            response = self.model.generate(prompt, max_new_tokens=256)
            pred = None
            if "answer" not in response:
                pred = "A"
            else:
                pattern = "answer"
                response = re.split(pattern, response, flags=re.IGNORECASE)[-1]
                for opt in options_chars:
                    if opt in response:
                        pred = opt
                        break
            if pred is None:
                pred = "A"

            self.predictions.append(pred)


class OpenQA(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output={}, id=d["id"]) 
                for d in task_data["data"]]

    def run_inference(self):
        self.predictions = []
        for inst in tqdm.tqdm(self.data):
            prompt = ""
            question = inst.input["question"]
            
            if "context" in inst.input.keys():
                context = inst.input["context"]
                prompt += f"Given the context: {context}\n"

            if self.task_data["task"] == "Temporal Reasoning":
                prompt += f"{question}\nAccroding to the provided context, how long does it take for the event? Please give a direct answer without other words"
            elif self.task_data["task"] == "Medical Question Answering":
                prompt += f"Please answer the question in a short pargraph: {question}"
            elif self.task_data["task"] == "Multilingual Question Answering":
                prompt += f"Please directly answer the question using the language in the question: {question}"
            elif self.task_data["task"] == "Table Question Answering":
                table = inst.input["table"]
                prompt += f"Please read the content of the table below carefully and then directly answer the question without other words:\n{table}\n\nQuestion: {question}\nAnswer:"
            else:
                prompt += f"Please directly answer the question in a short sentence: {question}"
                if self.task_data["task"] == "Document-Level Causal":
                    prompt += f"\nIf the context does not contain an answer to the question, simply output \"None of the above\"."
            
            response = self.model.generate(prompt, max_new_tokens=256)
            pred = response.strip()
            self.predictions.append(pred)


class SummarizationTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        instances = []
        for d in task_data["data"]:
            if "document_list" in d:
                instance = Instance(
                    input={"document_list": d["document_list"]},
                    output={},
                    id=d["id"]
                )
            elif d.get("input") and "highlights" in d.get("output", {}):
                instance = Instance(
                    input={"document": d["document"]},
                    output={},
                    id=d["id"]
                )
            else:
                instance = Instance(
                    input={"document": d["document"]},
                    output={},
                    id=d["id"]
                )
            instances.append(instance)
        return instances
        
    def run_inference(self):
        self.predictions = []
        for inst in tqdm.tqdm(self.data):
            if "document_list" in inst.input:
                doc_list = inst.input["document_list"]
                combined_docs = "\n".join(doc_list)
                
                prompt = (
                    "You are a multi-document summarization assistant.\n"
                    "Please read the following documents, and then summarize them in a concise paragraph:\n\n"
                    f"{combined_docs}\n\n"
                    "Summary:"
                )
            else:
                doc = inst.input["document"]
                prompt = (
                    "Please summarize the following document in a short sentence\n"
                    f"{doc}\n"
                    "Summary:"
                )

            pred = self.model.generate(prompt, max_new_tokens=256)

            if "Summary:" in pred:
                pred = pred.split("Summary:")[-1].strip()
            else:
                pred = pred.strip()
                
            self.predictions.append(pred)


class TranslationTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input={
                            "source_lang": d["in"], 
                            "target_lang": d["out"], 
                            "text": d["input"]
                         }, 
                         output={}, 
                         id=d["id"])
                for d in task_data["data"]]

    def run_inference(self):
        self.predictions = []
        for inst in tqdm.tqdm(self.data):
            source_lang = inst.input["source_lang"]
            target_lang = inst.input["target_lang"]
            text = inst.input["text"]

            prompt = (f"Please directly Translate the following text from {source_lang} to {target_lang}.\n"
                      f"Text: {text}\n"
                      f"Translation:")
            pred = self.model.generate(prompt, max_new_tokens=256)
            if "Translation:" in pred:
                pred = pred.split("Translation:")[-1].strip()
            else:
                pred = pred.strip()

            self.predictions.append(pred)


class StoryGenerationTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        instances = []
        for d in task_data["data"]:
            instances.append(
                Instance(
                    input=d["input"], 
                    output={},
                    id=d["id"]
                )
            )
        return instances

    def run_inference(self):
        self.predictions = []
        for inst in tqdm.tqdm(self.data):
            prompt_text = inst.input["prompt"]
            prompt = f"Please write a story based on the following prompt:\n{prompt_text}\nStory:"
            pred = self.model.generate(prompt, max_new_tokens=512)
            if "Story:" in pred:
                pred = pred.split("Story:")[-1].strip()

            self.predictions.append(pred)


class DialogueGenerationTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        instances = []
        for d in task_data["data"]:
            dialog_list = d.get("dialog", [])
            if not dialog_list:
                continue

            instances.append(
                Instance(
                    input={"dialog": dialog_list},
                    output={},
                    id=d["id"]
                )
            )
        return instances

    def run_inference(self):
        self.predictions = []

        for inst in tqdm.tqdm(self.data):
            dialog_context = inst.input["dialog"]
            prompt = "Below is a multi-turn conversation. Please continue the dialogue for the last turn.\n\n"
            for turn_idx, turn in enumerate(dialog_context):
                prompt += f"Turn {turn_idx + 1}: {turn}\n"
            prompt += "\nNow please respond in one short answer:\n"

            pred = self.model.generate(prompt, max_new_tokens=128).strip()
            self.predictions.append(pred)


class CodeGenerationTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        instances = []
        for d in task_data["data"]:
            instance_id = d["id"]
            language = d["language"]
            goal = d["goal"]
            context = d.get("context", [])

            instances.append(
                Instance(
                    input={
                        "language": language,
                        "goal": goal,
                        "context": context
                    },
                    output={},
                    id=instance_id
                )
            )
        return instances

    def run_inference(self):
        self.predictions = []
        self.languages = []

        for inst in tqdm.tqdm(self.data):
            language = inst.input["language"]
            goal = inst.input["goal"]
            context = inst.input["context"]

            prompt = f"You are an AI developer. Your goal is: {goal}\n"
            prompt += f"Please write {language} code that solves the described task.\n\n"

            for c_item in context:
                c_type = c_item["type"]
                c_content = c_item["content"]
                if c_type == "description":
                    prompt += f"Description:\n{c_content}\n\n"
                elif c_type == "example":
                    prompt += "Examples:\n"
                    for ex in c_content:
                        prompt += f"- Input: {ex['input']}, Expected Output: {ex['output']}\n"
                    prompt += "\n"
                else:
                    prompt += f"{c_type.capitalize()}:\n{c_content}\n\n"

            prompt += (
                "Now, please output ONLY the final code solution (without additional explanations, comments or text)."
                "\nCode:\n"
            )

            pred_code = self.model.generate(prompt, max_new_tokens=256).strip()
            if "Code:" in pred_code:
                pred_code = pred_code.split("Code:", 1)[-1].strip()

            self.predictions.append(pred_code)
            self.languages.append(language)


class CodeRepairTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        instances = []
        for d in task_data["data"]:
            instance_id = d["id"]
            input_part = d["input"]

            prompt = input_part["prompt"]
            source_code = input_part["sourceCode"]
            instances.append(
                Instance(
                    input={
                        "prompt": prompt,
                        "sourceCode": source_code
                    },
                    output={},
                    id=instance_id
                )
            )
        return instances

    def run_inference(self):
        self.predictions = []
        
        for inst in tqdm.tqdm(self.data):
            prompt = inst.input["prompt"]
            source_code = inst.input["sourceCode"]
            final_prompt = (
                f"{prompt}\n"
                f"{source_code}\n\n"
                "Now, please output ONLY the final code solution (without additional explanations, comments or text)."
                "Refined Code:"
            )

            pred_code = self.model.generate(final_prompt, max_new_tokens=256).strip()
            if "Refined Code:" in pred_code:
                pred_code = pred_code.split("Refined Code:", 1)[-1].strip()

            self.predictions.append(pred_code)


class CodeDefectDetectionTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        instances = []
        for d in task_data["data"]:
            instances.append(
                Instance(
                    input={"func": d["func"]},
                    output={},
                    id=d["id"]
                )
            )
        return instances

    def run_inference(self):
        self.predictions = []

        for inst in tqdm.tqdm(self.data):
            code_snippet = inst.input["func"]
            prompt = (
                "You are a code reviewer. Below is a piece of code or function:\n"
                f"{code_snippet}\n\n"
                "Please review carefully and determine if it contains a grammatical or logical defect. "
                "For example, the code below has defect:\n"
                "static void show_packets(AVFormatContext *format_ctx)\n\n{\n\n    AVPacket packet;\n\n\n\n    av_init_packet(&packet);\n\n    probe_array_header(\"packets\", 0);\n\n    while (!av_read_frame(format_ctx, &packet))\n\n        show_packet(format_ctx, &packet);\n\n    probe_array_footer(\"packets\", 0);\n\n}\n"
                "For another example, the code below has no defect:\n"
                "static void visitor_output_setup_internal(TestOutputVisitorData *output_data,\n\n                                          bool is_human)\n\n{\n\n    output_data->human = is_human;\n\n    output_data->sov = string_output_visitor_new(is_human);\n\n    g_assert(output_data->sov);\n\n    output_data->ov = string_output_get_visitor(output_data->sov);\n\n    g_assert(output_data->ov);\n\n}\n"
                "Output only 'No defect' if it does NOT contain a grammatical or logical defect, "
                "or ouput only 'Defect' if it DOES contain a defect.\n"
                "Answer:"
            )

            response = self.model.generate(prompt, max_new_tokens=16).strip()

            if "no defect" in response.lower():
                pred = "0"
            else:
                pred = "1"

            self.predictions.append(pred)


class TextToSQLTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        instances = []
        for d in task_data["data"]:
            instances.append(
                Instance(
                    input={
                        "context": d["input"]["context"],
                        "question": d["input"]["question"],
                    },
                    output={},
                    id=d["id"]
                )
            )
        return instances

    def run_inference(self):
        self.predictions = []

        for inst in tqdm.tqdm(self.data):
            schema_context = inst.input["context"]
            question = inst.input["question"]

            prompt = (
                "Below is a database schema:\n"
                f"{schema_context}\n"
                "Given the schema, please write a valid SQL query that answers the following question without other words.\n"
                f"Question: {question}\n"
                "SQL:"
            )

            response = self.model.generate(prompt, max_new_tokens=256)
            if "SQL:" in response:
                pred_sql = response.split("SQL:", 1)[-1].strip()
            else:
                pred_sql = response.strip()

            self.predictions.append(pred_sql)


class CodeExplanationTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        instances = []
        for d in task_data["data"]:
            code_snippet = d["code"]
            instance_id = d["id"]

            instances.append(
                Instance(
                    input={"code": code_snippet},
                    output={},
                    id=instance_id
                )
            )
        return instances

    def run_inference(self):
        self.predictions = []

        for inst in tqdm.tqdm(self.data):
            code_snippet = inst.input["code"]
            prompt = (
                "You are a code explainer. "
                "Please read the following code snippet and provide a concise, clear explanation in natural language:. For example:\n"
                "Code:\nboolean equalsResidueRing ( Object obj ) { if ( !( obj instanceof ResidueRing ) ) { return false ; } ResidueRing < C > otherRing = null ; try { otherRing = ( ResidueRing < C > ) obj ; } catch ( ClassCastException e ) { return false ; } if ( otherRing == null ) { return false ; } if ( ! ring . equals ( otherRing . ring ) ) { return false ; } return modul . equals ( otherRing . modul ) ; }"
                "Explanation: compares this ResidueRing with another object.\n\n"
                "Now please explain the code below without other words:\n"
                f"{code_snippet}\n"
                "Explanation:"
            )

            pred_explanation = self.model.generate(prompt, max_new_tokens=256).strip()
            self.predictions.append(pred_explanation)


class MathematicalProofGenerationTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        instances = []
        for d in task_data["data"]:
            statement = d["statement"]

            instances.append(
                Instance(
                    input={
                        "statement": statement
                    },
                    output={},
                    id=d["id"]
                )
            )
        return instances

    def run_inference(self):
        self.predictions = []

        for inst in tqdm.tqdm(self.data):
            statement = inst.input["statement"]

            prompt = (
                "You are a mathematical assistant. "
                "Please provide a clear, step-by-step proof for the following statement:\n"
                f"Statement: {statement}\n\n"
                "Ensure you include the final conclusion as well. Proof:"
            )

            pred_proof = self.model.generate(prompt, max_new_tokens=512).strip()
            self.predictions.append(pred_proof)


class MathematicalWordProblemSolvingTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        instances = []
        for d in task_data["data"]:
            problem_text = d["problem"]["text"]
            constraints = d["problem"].get("constraints", [])

            instances.append(
                Instance(
                    input={
                        "problem_text": problem_text,
                        "constraints": constraints
                    },
                    output={},
                    id=d["id"]
                )
            )
        return instances

    def run_inference(self):
        self.predictions_steps = []
        self.predictions_final = []

        for inst in tqdm.tqdm(self.data):
            problem_text = inst.input["problem_text"]
            constraints = inst.input["constraints"]
            constraints_str = ""
            if constraints:
                constraints_str = "\nConstraints:\n" + "\n".join(constraints)

            prompt = (
                "You are a math problem solver. Please solve the following word problem step by step. "
                "Finally, provide the final numeric or short answer in a separate line labeled as 'Final Answer:'.\n\n"
                f"Problem:\n{problem_text}{constraints_str}\n\n"
                "Solution (step-by-step) + Final Answer:\n"
            )

            response = self.model.generate(prompt, max_new_tokens=512).strip()

            steps_part, final_part = response, ""
            if "Final Answer:" in response:
                parts = response.split("Final Answer:", 1)
                steps_part = parts[0].strip()
                final_part = parts[1].strip()

            self.predictions_steps.append(steps_part)
            self.predictions_final.append(final_part)


class ParaphraseGenerationTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        instances = []
        for d in task_data["data"]:
            instances.append(
                Instance(
                    input={"originalSentence": d["input"]["originalSentence"]},
                    output={},
                    id=d["id"]
                )
            )
        return instances

    def run_inference(self):
        self.predictions = []
        for inst in tqdm.tqdm(self.data):
            original_sentence = inst.input["originalSentence"]
            
            prompt = (
                "Please rewrite the following sentence in a different way but keep the same meaning:\n"
                f"{original_sentence}\n"
                "Paraphrase:"
            )

            pred = self.model.generate(prompt, max_new_tokens=128)

            if "Paraphrase:" in pred:
                pred = pred.split("Paraphrase:")[-1].strip()

            self.predictions.append(pred.strip())


class GrammarCorrectionTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [
            Instance(
                input=d["input"],
                output={},
                id=d["id"]
            )
            for d in task_data["data"]
        ]
    
    def run_inference(self):
        self.predictions = []

        for inst in tqdm.tqdm(self.data):
            error_type = inst.input["Error Type"]
            ungrammatical_sentence = inst.input["Ungrammatical Statement"]
            
            prompt = (
                f"You are a grammar correction assistant.\n"
                f"There is a sentence with the following error type: {error_type}.\n"
                f"Please rewrite the sentence in correct standard English without any other word.\n\n"
                f"Ungrammatical Sentence: {ungrammatical_sentence}\n\n"
                f"Rewritten Sentence:"
            )

            corrected = self.model.generate(prompt, max_new_tokens=128).strip()
            if "Rewritten Sentence:" in corrected:
                corrected = corrected.split("Rewritten Sentence:")[-1].strip()

            self.predictions.append(corrected)


class TextStyleTransferTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        instances = []
        for d in task_data["data"]:
            instances.append(
                Instance(
                    input={
                        "text": d["input"]["text"],
                        "style": d["input"]["style"]
                    },
                    output={},
                    id=d["id"]
                )
            )
        return instances

    def run_inference(self):
        self.predictions = []

        for inst in tqdm.tqdm(self.data):
            text = inst.input["text"]
            style = inst.input["style"]

            prompt = (
                "You are a style transfer assistant.\n"
                "Below is a piece of text and a target style.\n"
                f"Text: {text}\n"
                f"Style: {style}\n\n"
                "Please rewrite the above text to match the target style more accurately, "
                "while keeping the original meaning intact.\n"
                "Answer:"
            )
            
            pred = self.model.generate(prompt, max_new_tokens=256).strip()
            if "Answer:" in pred:
                pred = pred.split("Answer:")[-1].strip()
            
            self.predictions.append(pred)


class TableToTextGenerationTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        instances = []
        for d in task_data["data"]:
            instance_id = d["id"]
            table_data = d["input"]["table"]
            instances.append(
                Instance(
                    input={"table": table_data},
                    output={},
                    id=instance_id
                )
            )
        return instances

    def run_inference(self):
        self.predictions = []

        for inst in tqdm.tqdm(self.data):
            table_data = inst.input["table"]

            prompt = "Below is a table. Please generate a coherent description that summarizes the table's content.\n\n"
            for table_idx, table_item in enumerate(table_data):
                header = table_item["header"]
                rows = table_item["rows"]
                prompt += f"Table {table_idx+1}:\nHeader: {header}\nRows:\n"
                for r_idx, row in enumerate(rows):
                    prompt += f"{r_idx+1}. {row}\n"
                prompt += "\n"

            prompt += "Now write a concise text describing the above table:\n"

            pred_text = self.model.generate(prompt, max_new_tokens=512)
            pred_text = pred_text.strip()

            self.predictions.append(pred_text)


class TimeSeriesForecastingTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        instances = []
        for d in task_data["data"]:
            time_series = d["input"]["data"]
            instances.append(
                Instance(
                    input={"time_series": time_series},
                    output={},
                    id=d["id"]
                )
            )
        return instances

    def run_inference(self):
        self.predictions = []
        for inst in tqdm.tqdm(self.data):
            series_data = inst.input["time_series"]
            pred_values = self.model.generate_for_timeseries(series_data, horizon=1, freq=0)
            predicted = pred_values[0] if pred_values else 0.0
            self.predictions.append(predicted)


class ClassificationTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output={}, id=d["id"]) 
                for d in task_data["data"]]
    
    def run_inference(self):
        self.predictions = []
        for inst in tqdm.tqdm(self.data):
            if 'stance_detection' in self.task_data['task']:
                tweets = inst.input["tweets"]
                target = inst.input["target"]
                prompt = inst.input["prompt"].replace("<<<target>>>", target).replace("<<<tweets>>>", tweets)
            elif 'aspect_sentiment_classification' in self.task_data['task']:
                raw_text = inst.input["raw_text"]
                target = inst.input["target"]
                prompt = inst.input["prompt"].replace("<<<raw_text>>>", raw_text).replace("<<<target>>>", target) + 'Please direct return the category name without any other words.'
            elif 'target_oriented_opinion_words_extraction' in self.task_data['task']:
                raw_text = inst.input["raw_text"]
                aspect = inst.input["aspect"]
                prompt = inst.input["prompt"].replace("<<<raw_text>>>", raw_text).replace("<<<aspect>>>", aspect) + 'Please direct return the opinion word without any other words.'
            else:
                raw_text = inst.input["raw_text"]
                prompt = inst.input["prompt"].replace("<<<raw_text>>>", raw_text) + 'Please return the desired result directly, without any other explanation.'
            response = self.model.generate(prompt, max_new_tokens=64)
            self.predictions.append(response.lower())


class MultiLabelClassificationTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output={}, id=d["id"]) 
                for d in task_data["data"]]
    
    def run_inference(self):
        self.predictions = []
        for inst in tqdm.tqdm(self.data):
            raw_text = inst.input["raw_text"]
            prompt = inst.input["prompt"].replace("<<<raw_text>>>", raw_text)
            prompt = prompt + " Please return the desired result directly, without any other explanation." + " Split the result by commas instead of \\n."
            response = self.model.generate(prompt, max_new_tokens=64)
            self.predictions.append('<p>'.join(response.lower().split(', ')))


class ChoiceTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output={}, id=d["id"]) 
                for d in task_data["data"]]
    
    def run_inference(self):
        self.predictions = []
        for inst in tqdm.tqdm(self.data):
            raw_text = inst.input["raw_text"]
            prompt = inst.input["prompt"].replace("<<<raw_text>>>", raw_text) + 'Please return the desired result directly, without any other explanation.'
            response = self.model.generate(prompt, max_new_tokens=64)
            if len(response.strip()) > 1:
                if "A" in response.strip():
                    response = "A"
                elif "B" in response.strip():
                    response = "B"
                elif "C" in response.strip():
                    response = "C"
                elif "D" in response.strip():
                    response = "D"
            self.predictions.append(response.lower())


class NERTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output={}, id=d["id"]) 
                for d in task_data["data"]]
    
    def run_inference(self):
        self.predictions = []
        for inst in tqdm.tqdm(self.data):
            text = inst.input["raw_text"]
            prompt = inst.input["prompt"].replace("<<<raw_text>>>", text)
            response = self.model.generate(prompt, max_new_tokens=128)
            self.predictions.append('<p>'.join(response.lower().split(', ')))


def save_predictions(task_obj: BaseTask, task_directory: str):
    save_path = os.path.join(task_directory, "prediction.json")
    records = []
    if isinstance(task_obj, MathematicalWordProblemSolvingTask):
        for idx, inst in enumerate(task_obj.data):
            records.append({
                "id": inst.id,
                "prediction_steps": task_obj.predictions_steps[idx],
                "prediction_final": task_obj.predictions_final[idx]
            })
    elif isinstance(task_obj, TimeSeriesForecastingTask):
        for idx, inst in enumerate(task_obj.data):
            records.append({
                "id": inst.id,
                "prediction": float(task_obj.predictions[idx])
            })
    else:
        for idx, inst in enumerate(task_obj.data):
            pred_val = task_obj.predictions[idx]
            if isinstance(pred_val, (np.floating, np.integer)):
                pred_val = float(pred_val)
            records.append({"id": inst.id, "prediction": pred_val})
    with open(save_path, "w", encoding="utf-8") as fp:
        json.dump(records, fp, ensure_ascii=False, indent=2)


TASK_MAPPING = {
    "MultipleChoiceQA": MultipleChoiceQA,
    "OpenQA": OpenQA,
    "Summarization": SummarizationTask,
    "Story Generation": StoryGenerationTask,
    "Translation": TranslationTask,
    "Dialogue": DialogueGenerationTask,
    "Code Generation": CodeGenerationTask,
    "Code Defect Detection": CodeDefectDetectionTask,
    "Code Repair": CodeRepairTask,
    "Code Explanation": CodeExplanationTask,
    "Proof": MathematicalProofGenerationTask,
    "Mathematical Word Problem Solving": MathematicalWordProblemSolvingTask,
    "Text to SQL": TextToSQLTask,
    "Paraphrase Generation": ParaphraseGenerationTask,
    "Grammar Correction": GrammarCorrectionTask,
    "Table-to-Text Generation": TableToTextGenerationTask,
    "Time Series": TimeSeriesForecastingTask,
    "Text Style Transfer": TextStyleTransferTask,
    "classification": ClassificationTask,
    "multi label classification": MultiLabelClassificationTask,
    "ner": NERTask,
    "extraction": MultiLabelClassificationTask,
    "relation extraction": MultiLabelClassificationTask,
    "event detection": MultiLabelClassificationTask,
    "parsing": MultiLabelClassificationTask,
    "multiple choice": ChoiceTask,
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NLP Predictor")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    data_root = os.path.abspath(args.dataset_dir)
    model = LLMModel(args.model_name)

    task_dirs = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

    for idx, task_folder in enumerate(task_dirs, start=1):
        folder_path = os.path.join(data_root, task_folder)
        annotation_path = os.path.join(folder_path, "annotation.json")

        with open(annotation_path, "r", encoding="utf-8") as f:
            task_data = json.load(f)

        task_type = task_data.get("type")
        task_name = task_data.get("task", task_folder)
        print(f"\nTask {idx}/{len(task_dirs)}: {task_name} (Type = {task_type})")

        task_class = TASK_MAPPING.get(task_type, OpenQA)
        task = task_class(task_data, model)

        task.run_inference()
        save_predictions(task, folder_path)
