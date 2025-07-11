from email.mime import audio
import json
import os
from pandas import read_json
from regex import B, D
import tqdm
from typing import List, Dict, Any
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dataclasses import dataclass
from abc import ABC, abstractmethod
from rouge_score import rouge_scorer
import math
import time
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def read_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def exact_match_accuracy(predictions: List[str], references: List[str]) -> float:
    correct = 0
    for pred, ref in zip(predictions, references):
        if isinstance(ref, str):
            ref = [ref]
        if isinstance(ref, int):
            ref = [ref]
        is_match_this_turn = False
        for r in ref:
            if pred.strip() == r.strip():
                is_match_this_turn = True
        if is_match_this_turn:
            correct += 1
    return correct / len(predictions) if predictions else 0.0


def blur_match_accuracy(predictions: List[str], references: List[str]) -> float:
    correct = 0
    for pred, ref in zip(predictions, references):
        # if isinstance(ref, int):
        #     if  == ref:
        if str(ref) in str(pred).strip().lower():
            correct += 1
    return correct / len(predictions) if predictions else 0.0


def calculate_f1(predictions: List[str], references: List[str]) -> float:
    def compute_f1(pred: str, ref: str) -> float:
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()
        
        common_tokens = set(pred_tokens) & set(ref_tokens)
        num_common = len(common_tokens)
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        
        return 2 * precision * recall / (precision + recall)
    
    total_f1 = 0.0
    for pred, ref in zip(predictions, references):
        if isinstance(ref, str):
            ref = [ref]
        max_f1 = 0.0
        for r in ref:
            max_f1 = max(compute_f1(pred, r), max_f1)
        total_f1 += max_f1
    
    return total_f1 / len(predictions) if predictions else 0.0


def rouge_evaluation(predictions: List[str], references: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougel_scores = [], [], []
    for pred, ref in zip(predictions, references):
        if isinstance(ref, str):
            ref = [ref]
        rouge1, rouge2, rougeL = 0, 0, 0
        for r in ref:
            scores = scorer.score(r, pred)
            rouge1 = max(scores['rouge1'].fmeasure, rouge1)
            rouge2 = max(scores['rouge2'].fmeasure, rouge2)
            rougeL = max(scores['rougeL'].fmeasure, rougeL)
        rouge1_scores.append(rouge1)
        rouge2_scores.append(rouge2)
        rougel_scores.append(rougeL)
    return {
        'rouge1': sum(rouge1_scores) / len(rouge1_scores),
        'rouge2': sum(rouge2_scores) / len(rouge2_scores),
        'rougeL': sum(rougel_scores) / len(rougel_scores),
    }


def bleu_evaluation(predictions: List[str], references: List[str]) -> Dict[str, float]:
    smoothie = SmoothingFunction().method4
    bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
    
    for pred, ref in zip(predictions, references):
        hypothesis = nltk.word_tokenize(pred)
        if isinstance(ref, str):
            ref = [ref]
        bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
        for r in ref:
            reference = [nltk.word_tokenize(r)]
            bleu1 = max(sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothie), bleu1)
            bleu2 = max(sentence_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie), bleu2)
            bleu3 = max(sentence_bleu(reference, hypothesis, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothie), bleu3)
            bleu4 = max(sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie), bleu4)
        
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


def mean_absolute_error(predictions: List[float], references: List[float]) -> float:
    if not predictions:
        return 0.0
    error_sum = 0.0
    for p, r in zip(predictions, references):
        error_sum += abs(p - r)
    return error_sum / len(predictions)


def mean_squared_error(predictions: List[float], references: List[float]) -> float:
    if not predictions:
        return 0.0
    error_sum = 0.0
    for p, r in zip(predictions, references):
        error_sum += (p - r) ** 2
    return error_sum / len(predictions)


def root_mean_squared_error(predictions: List[float], references: List[float]) -> float:
    return math.sqrt(mean_squared_error(predictions, references))


def post_process_output(output: str) -> str:
    cnt = 0
    for d in output:
        if d['gt'] in d['response'].strip().lower():
            cnt += 1
    acc = round(cnt / len(output), 4)
    print(f"Accuracy: {acc}")
    return acc


def evaluation_accuracy(predictions: List[str]) -> Dict[str, float]:
    correct = 0
    for pred in predictions:
        if pred == '1':
            correct += 1
    return correct / len(predictions) if predictions else 0.0


class AudioComprehensionModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.load_model()
    
    def load_model(self):
        if 'qwen-audio-chat' in self.model_name.lower():
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='cuda', trust_remote_code=True).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token_id = self.tokenizer.eod_id
        elif 'qwen2' in self.model_name.lower():
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            print(self.processor.chat_template)
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(self.model_name, device_map="auto").eval()
        
        elif 'new_model_name' in self.model_name.lower():
            # support to load self-build models here
            pass

        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
    def generate(self, prompt: str, max_new_tokens=256, audio_path: str=None) -> str:
        
        if "qwen-audio-chat" in self.model_name.lower():
            query = self.tokenizer.from_list_format([
                {'audio': audio_path}, # Either a local path or an url
                {'text': prompt} # The query,
            ])
            response, history = self.model.chat(self.tokenizer, query=query, history=None)
            return response
        
        elif "qwen2" in self.model_name.lower():
            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'}, 
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt},
                ]},
            ]
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audios.append(
                                librosa.load(
                                    ele['audio'], 
                                    sr=self.processor.feature_extractor.sampling_rate)[0]
                            )
            # print(text)
            inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True)
            inputs.input_ids = inputs.input_ids.to("cuda")
            inputs = inputs.to("cuda")
            # print(inputs)
            # exit(0)
            generate_ids = self.model.generate(**inputs, max_length=300)
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]

            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return response
        
        elif "new" in self.model_name.lower():
            # support to generate response based on self-build models here
            pass
        
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        


@dataclass
class Instance:
    input: Dict[str, Any]
    output: Dict[str, Any]
    id: str


class BaseTask(ABC):
    def __init__(self, task_data: Dict[str, Any], model: AudioComprehensionModel, audio_dir: str = None, output_dir: str = None, task_name: str = None):
        self.task_data = read_json(task_data)
        self.model = model
        self.audio_dir = audio_dir  # should include the audios files
        self.data = self._parse_data(self.task_data)
        self.choice_candidate = self._get_choice_candidate(self.task_data)
        self.task_name = os.path.dirname(task_data).split("/")[-1] if task_name is None else task_name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True) if self.output_dir else None

        self.references = []
        self.predictions = []

    def save_predictions(self, audio_paths):
        results = []
        for gt, response, audio_path in zip(self.references, self.predictions, audio_paths):
            results.append({
                'gt': gt,
                'response': response,
                'audio_path': audio_path,
            })
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = os.path.join(self.output_dir, f'{self.task_name }_{time_prefix}.json') if self.output_dir else f'{self.task_name }_{time_prefix}.json'
        json.dump(results, open(results_file, 'w'))

    @abstractmethod
    def _get_choice_candidate(self):
        pass

    @abstractmethod
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def run_inference(self):
        pass


class EvaluationTask(BaseTask):
    """
    Used to determine whether the results generated by the model are correct
    """
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return task_data

    def _get_choice_candidate(self, data: List[Instance]) -> List[str]:
        return ["None"]

    def save_predictions(self, audio_paths):
        results = []
        for gt, response, audio_path in zip(self.references, self.predictions, audio_paths):
            results.append({
                'gt': gt[0],
                'response': gt[1],
                'audio_path': audio_path,
                'llm_prediction': response,
            })
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = os.path.join(self.output_dir, f'{self.task_name }_{time_prefix}.json') if self.output_dir else f'{self.task_name }_{time_prefix}.json'
        json.dump(results, open(results_file, 'w'))

    def run_inference(self):
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            prompt = " will provide you with a Ground-truth label and a Prediction label. The label can either be a single string or a list of multiple labels. I need you to compare these two labels on a semantic level.\nSpecifically, I want you to evaluate whether the Prediction label semantically matches, is partially aligned, includes, or describes the Ground-truth label (or the semantic meaning represented by the list of labels). If any of these conditions are satisfied, consider it a match.\n\nHere are some examples of successful matches:\n\nGround-truth label: \"rain\"\nPrediction label: \"The sound in the audio is rain falling\"\n(This is considered a match.)\nGround-truth label: [\"decrease\", \"volume\", \"none\"]\nPrediction label: \"The intent in the audio is to adjust the volume\"(This is also considered a match.)\nIf the labels successfully match, assign a score of 1. If they do not match, assign a score of 0.**Imporant!!!, only output the score (0 or 1), no explanation.** \n\nGround-truth label:{}\nPrediction label:{}"
            gt = inst["gt"]
            response = inst["response"]
            prompt = prompt.format(gt, response)
            try:
                response = self.model.generate(prompt)
                # print(response)
            except Exception as e:
                response = "None"
                continue

            self.predictions.append(response)
            self.references.append([inst["gt"], inst["response"]])
            audio_paths.append(inst["audio_path"])
        self.save_predictions(audio_paths)

    def evaluate(self) -> Dict[str, float]:
        acc = evaluation_accuracy(self.predictions)
        return {"accuracy": acc}


class AccentSexClassification(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data: List[Instance]) -> List[str]:
        return ['female', 'male']

    def save_predictions(self, audio_paths):
        results = []
        for gt, response, audio_path in zip(self.references, self.predictions, audio_paths):
            results.append({
                'gt': gt,
                'response': response,
                'audio_path': audio_path,
            })
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = os.path.join(self.output_dir, f'{self.task_name }_{time_prefix}.json') if self.output_dir else f'{self.task_name }_{time_prefix}.json'
        json.dump(results, open(results_file, 'w'))

    def run_inference(self):
        self.predictions = []
        self.references = []
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            question = inst.input["prompt"]
            prompt = f"Please listen to the audio and then answer the question by directly choose a choice from choice candidates. Questions: {question}, Candidate choices: {self.choice_candidate}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except:
                print("error audio {}".format(inst.input["audio_file"]))
                continue
            self.predictions.append(response)
            self.references.append(inst.output["text"])
            audio_paths.append(inst.input["audio_file"])
        
        self.save_predictions(audio_paths)
    
    
    def evaluate(self) -> Dict[str, float]:
        acc = exact_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}


class AcousticSceneClassification(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data: List[Instance]) -> List[str]:
        choices = []
        for item in data['data']:
            choices.append(item['output']["text"].strip().lower())
        choices = list(set(choices))
        return choices

    def run_inference(self):
        print(f"Choice candidates: {self.choice_candidate}")
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            question = inst.input["prompt"]
            prompt = f"Please listen to the input music and then determine the category of the acoustic scene. The candidate scene category are {self.choice_candidate}. Please output **only one category** from the provided candidate categories, and **DO NOT** output any other words.\nQuestions: {question}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(inst.output["text"].strip().lower())
            audio_paths.append(inst.input["audio_file"])
        self.save_predictions(audio_paths)
    
    def evaluate(self) -> Dict[str, float]:
        acc = exact_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}


class AnimalSoundDetection(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data) -> List[str]:
        choices = []
        for item in data['data']:
            choices.append(item['output']["text"].strip().lower())
        choices = list(set(choices))
        return choices

    def run_inference(self):
        print(f"Choice candidates: {self.choice_candidate}")
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            question = inst.input["prompt"]
            prompt = f"Please listen to the audio and then answer the question by directly choose a choice from choice candidates, without other words. Questions: {question}, Candidate choices: {self.choice_candidate}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(inst.output["text"].strip().lower())
            audio_paths.append(inst.input["audio_file"])
        self.save_predictions(audio_paths)

    def evaluate(self) -> Dict[str, float]:
        acc = exact_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}


class AudioCaptions(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data: List[Instance]) -> List[str]:
        return ["None"]

    def run_inference(self):
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            question = inst.input["prompt"]
            prompt = f"Please listen to the audio and then answer the question. Questions: {question}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(inst.output["text"])
            audio_paths.append(inst.input["audio_file"])
        self.save_predictions(audio_paths)

    def evaluate(self) -> Dict[str, float]:
        bleu = bleu_evaluation(self.predictions, self.references)
        return {"bleu1": bleu['bleu1']}


class AudioCaptionsClotho(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data: List[Instance]) -> List[str]:
        return ["None"]

    def run_inference(self):
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            question = inst.input["prompt"]
            prompt = f"Please listen to the audio and then answer the question. Questions: {question}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(inst.output["text"])
            audio_paths.append(inst.input["audio_file"])
        self.save_predictions(audio_paths)

    def evaluate(self) -> Dict[str, float]:
        acc = bleu_evaluation(self.predictions, self.references)
        return {"accuracy": acc}


class AudioQA(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data) -> List[str]:
        choices = []
        for item in data['data']:
            choices.append(item['output']["text"].strip().lower())
        choices = list(set(choices))
        return choices

    def run_inference(self):
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            question = inst.input["prompt"]
            prompt = f"Please listen to the audio and then answer the question by directly choose a choice from choice candidates. Questions: {question}, Candidate choices: {self.choice_candidate}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(inst.output["text"])
            audio_paths.append(inst.input["audio_file"])
        self.save_predictions(audio_paths)

    def evaluate(self) -> Dict[str, float]:
        acc = exact_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}


class BirdSoundDetection(BaseTask):

    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data: List[Instance]) -> List[str]:
        return ["Yes", "No"]

    def save_predictions(self, audio_paths):
        results = []
        for gt, response, audio_path in zip(self.references, self.predictions, audio_paths):
            results.append({
                'gt': gt,
                'response': response,
                'audio_path': audio_path,
            })
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = os.path.join(self.output_dir, f'{self.task_name }_{time_prefix}.json') if self.output_dir else f'{self.task_name }_{time_prefix}.json'
        json.dump(results, open(results_file, 'w'))

    def run_inference(self):
        self.predictions = []
        self.references = []
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            question = inst.input["prompt"]
            prompt = f"Please listen to the audio and then answer the question by directly choose a choice from choice candidates. Questions: {question}, Candidate choices: {self.choice_candidate}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append("Yes" if inst.output["text"] == 1 else "No")
            audio_paths.append(inst.input["audio_file"])
        self.save_predictions(audio_paths)

    def evaluate(self) -> Dict[str, float]:
        acc = exact_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}


class EnvironmentSoundRecognition(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data) -> List[str]:
        choices = []
        for item in data['data']:
            choices.append(item['output']["text"].strip().lower())
        choices = list(set(choices))
        return choices

    def run_inference(self):
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            question = inst.input["prompt"]
            prompt = f"Please listen to the audio and then answer the question by directly choose a choice from choice candidates. Questions: {question}, Candidate choices: {self.choice_candidate}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print(f"error {e}")
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(inst.output["text"])
            audio_paths.append(inst.input["audio_file"])
        self.save_predictions(audio_paths)
    
    def evaluate(self) -> Dict[str, float]:
        acc = blur_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}


class IntentClassification(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data: Dict) -> Dict:
        intent_label = data['intent_label']
        return intent_label

    def run_inference(self):
        audio_paths = []
        candidate_actions = ','.join([k for k in self.choice_candidate['action'].keys() if not k[0].isdigit()])
        candidate_objects = ','.join([k for k in self.choice_candidate['object'].keys() if not k[0].isdigit()])
        candidate_locations = ','.join([k for k in self.choice_candidate['location'].keys() if not k[0].isdigit()])
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            question = inst.input["prompt"]
            prompt = f"Please listen to the audio and then detect the intention. The intention triplet includes three parts: action, object, and location. The candicate actions are {candidate_actions}, candidate objects are {candidate_objects}, and candidate locations are {candidate_locations}. Please answer the questions only use the provided candidate actions, objects, and locations. Questions: {question}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(' '.join([self.choice_candidate['action'][inst.output["text"].split()[0]], self.choice_candidate['action'][inst.output["text"].split()[1]], self.choice_candidate['action'][inst.output["text"].split()[2]]]))
            audio_paths.append(inst.input["audio_file"])
        self.save_predictions(audio_paths)

    def evaluate(self) -> Dict[str, float]:
        acc = exact_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}


def post_process_intent_output():
    data_path = '/m2v_intern/wushengqiong/model/audio-test/predictions/understanding/IntentClassification_250102204424.json'
    intent_label = read_json('/m2v_intern/wushengqiong/model/audio-test/understanding/IntentClassification/annotation.json')['intent_label']
    action = intent_label['action']
    object = intent_label['object']
    location = intent_label['location']

    data = read_json(data_path)

    results = []
    for d in data:
        results.append({
            'gt': [action[d['gt'].split()[0]], object[d['gt'].split()[1]], location[d['gt'].split()[2]]],
            'response': d['response'],
            'audio_path': d['audio_path'],
        })
    json.dump(results, open('/m2v_intern/wushengqiong/model/audio-test/predictions/understanding/IntentClassification_250102204424_1.json', 'w'))


class MusicGenreClassification(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data: Dict) -> Dict:
        choices = []
        for item in data['data']:
            choices.append(item['output']["text"].strip().lower())
        choices = list(set(choices))
        return choices


    def run_inference(self):
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"].replace('\\', '/'))
            question = inst.input["prompt"]
            prompt = f"Please listen to the input music and then determine the genre of the music. The candidate genres are {self.choice_candidate}. Please output **only one genre** from the provided candidate genres, and **DO NOT** output any other words.\nQuestions: {question}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(inst.output["text"])
            audio_paths.append(inst.input["audio_file"])
        self.save_predictions(audio_paths)

    def evaluate(self) -> Dict[str, float]:
        acc = exact_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}


class MusicInstrumentClassification(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data: Dict) -> Dict:
        choices = []
        for item in data['data']:
            choices.append(item['output']["text"].strip().lower())
        choices = list(set(choices))
        return choices

    def run_inference(self):
        audio_paths = []
        # candidate_instruments = ','.join([k for k in self.choice_candidate.keys() if not k[0].isdigit()])
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            question = inst.input["prompt"]
            prompt = f"Please listen to the music and then detect the instrument of the music. The candidate instruments are {self.choice_candidate}. Please output **only the most appropriate music instrument** from the provided candidate music instruments, and **DO NOT** output any other words. Questions: {question}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(inst.output["text"])
            audio_paths.append(inst.input["audio_file"])
        self.save_predictions(audio_paths)

    def evaluate(self) -> Dict[str, float]:
        acc = exact_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}


class MusicInstrumentSourceAnalysis(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data: Dict) -> Dict:
        choices = []
        for item in data['data']:
            choices.append(item['output']["text"].strip().lower())
        choices = list(set(choices))
        return choices

    def run_inference(self):
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            question = inst.input["prompt"]
            prompt = f"Please listen to the music and then detect the instrucment source of the music. The candidate sources are {self.choice_candidate}. Please output **only the most appropriate music source** from the provided candidate music sources, and **DO NOT** output any other words. Questions: {question}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(inst.output["text"])
            audio_paths.append(inst.input["audio_file"].strip().lower())
        self.save_predictions(audio_paths)

    def evaluate(self) -> Dict[str, float]:
        acc = exact_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}


class MusicPitchAnalysis(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data: Dict) -> Dict:
        choices = []
        for item in data['data']:
            choices.append(item['output']["text"])
        choices = list(set(choices))
        return choices

    def run_inference(self):
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            question = inst.input["prompt"]
            prompt = f"Please listen to the music and then detect the pitch score of the music. The 0-based MIDI pitch is in the range [0, 127]. Please output **only the most appropriate pitch score in a number** from the provided range, and **DO NOT** output any other words. Questions: {question}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(inst.output["text"])
            audio_paths.append(inst.input["audio_file"].strip().lower())
        self.save_predictions(audio_paths)
    
    def evaluate(self) -> Dict[str, float]:
        acc = exact_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}
    

class NoteQualitiesAnalysis(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data: Dict) -> Dict:
        choices = []
        for item in data['data']:
            choices.append(','.join(item['output']["text"]).strip().lower())
        choices = list(set(choices))
        return choices

    def run_inference(self):
        audio_paths = []
        for inst in tqdm.tqdm(self.data):  
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            question = inst.input["prompt"]
            prompt = f"Please listen to the music and then detect the note quality of the given music. The candidate annotation is {self.choice_candidate}. Please output **the qualities which are present in this note** from the provided candidate music note quality candidate categories, and **DO NOT** output any other words. Questions: {question}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(','.join(inst.output["text"]))
            audio_paths.append(inst.input["audio_file"].strip().lower())
        self.save_predictions(audio_paths)
    
    def evaluate(self) -> Dict[str, float]:
        acc = exact_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}


class OpenAQA(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data: Dict) -> Dict:
        choices = []
        for item in data['data']:
            choices.append(item['output']["text"].strip().lower())
        choices = list(set(choices))
        return choices

    def run_inference(self):
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            question = inst.input["prompt"]
            prompt = f"Please listen to the audio and then answer the question. Questions: {question}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(inst.output["text"])
            audio_paths.append(inst.input["audio_file"])
        self.save_predictions(audio_paths)

    def evaluate(self) -> Dict[str, float]:
        acc = bleu_evaluation(self.predictions, self.references)
        return {"accuracy": acc}


class SoundEventClassification(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data: Dict) -> Dict:
        choices = []
        for item in data['data']:
            choices.append(item['output']["text"].strip().lower())
        choices = list(set(choices))
        return choices

    def run_inference(self):
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            question = inst.input["prompt"]
            prompt = f"Please listen to the music and then detect the happening event of the given audio. The candidate annotation is {self.choice_candidate}. Please output **only one event** from the provided candidate events,, and **DO NOT** output any other words. Questions: {question}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(inst.output["text"])
            audio_paths.append(inst.input["audio_file"])
        self.save_predictions(audio_paths)

    def evaluate(self) -> Dict[str, float]:
        acc = exact_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}


class SpeechCommand(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data: Dict) -> Dict:
        choices = []
        for item in data['data']:
            choices.append(item['output']["text"].strip().lower())
        choices = list(set(choices))
        return choices

    def run_inference(self):
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"].replace('\\', '/'))
            question = inst.input["prompt"]
            prompt = f"Please listen to the audio and then detect the speech command of the given audio. The candidate annotation is {self.choice_candidate}. Please output **only one command** from the provided candidate commands, and **DO NOT** output any other words. Questions: {question}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(inst.output["text"].strip().lower())
            audio_paths.append(inst.input["audio_file"])
        self.save_predictions(audio_paths)

    def evaluate(self) -> Dict[str, float]:
        acc = exact_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}


class SpeechEmotionRecognition(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data: Dict) -> Dict:
        choices = []
        for item in data['data']:
            choices.append(item['output']["text"].strip().lower())
        choices = list(set(choices))
        return choices

    def run_inference(self):
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            question = inst.input["prompt"]
            prompt = f"Please listen to the audio and then detect the emotion of the given audio. The candidate annotation is {self.choice_candidate}. Please output **only one emotion** from the provided candidate emotions, and **DO NOT** output any other words. Questions: {question}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(inst.output["text"].strip().lower())
            audio_paths.append(inst.input["audio_file"])
        self.save_predictions(audio_paths)

    def evaluate(self) -> Dict[str, float]:
        acc = exact_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}


class VocalSoundClassification(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data: Dict) -> Dict:
        choices = []
        for item in data['data']:
            choices.append(item['output']["text"].strip().lower())
        choices = list(set(choices))
        return choices

    def run_inference(self):
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            question = inst.input["prompt"]
            prompt = f"Please listen to the audio and then detect the vocal sound category of the given audio. The candidate annotation is {self.choice_candidate}. Please output **only one vocal sound category** from the provided candidate vocal sounds, and **DO NOT** output any other words. Questions: {question}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(inst.output["text"].strip().lower())
            audio_paths.append(inst.input["audio_file"])
        self.save_predictions(audio_paths)

    def evaluate(self) -> Dict[str, float]:
        acc = exact_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}


class VocalTechniqueDetection(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def _get_choice_candidate(self, data: Dict) -> Dict:
        choices = []
        for item in data['data']:
            choices.append(item['output']["text"].strip().lower())
        choices = list(set(choices))
        return choices

    def run_inference(self):
        audio_paths = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"].replace('\\', '/'))
            question = inst.input["prompt"]
            prompt = f"Please listen to the audio and then detect the vocal technique of the given audio. The candidate annotations are scales, arpeggios, long tones, and excerpts. Please output **only one vocal technique** from the provided candidate vocal techniques, and **DO NOT** output any other words. Questions: {question}\nAnswer:"
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except Exception as e:
                print("Error audio: {}".format(inst.input["audio_file"]))
                response = "None"
                continue
            self.predictions.append(response)
            self.references.append(inst.output["text"].strip().lower())
            audio_paths.append(inst.input["audio_file"])
        self.save_predictions(audio_paths)

    def evaluate(self) -> Dict[str, float]:
        acc = exact_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}


def log_performance_csv(model_name, task_name, metric, score, root_path, output_file='prediction.json'):
    import csv
    file_exists = os.path.isfile(os.path.join(root_path, output_file))

    row_data = {
        'model': model_name,
        'task': task_name,
        'metric': metric,
        'score': str(score),
    }

    with open(os.path.join(root_path, output_file), mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()

        writer.writerow(row_data)


def log_performance_json(model_name, task_name, metric, score, root_path, output_file='prediction.json'):
    import json
    log_data = {
        'model': model_name,
        'task': task_name,
        'metric': metric,
        'score': str(score),
    }
    
    log_file_path = os.path.join(root_path, output_file)
    
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(log_data)

    with open(log_file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4)
    

def log_performance_detail(model_name, task_name, metrics, root_path, output_file='performance_log.csv'):
    import csv
    file_path = os.path.join(root_path, output_file)
    file_exists = os.path.isfile(file_path)
    
    # Retrieve the main indicator values from the metrics dictionary
    metric_value = None
    if isinstance(metrics, dict):
        # Select metrics based on priority
        for key in ['accuracy', 'f1', 'micro_f1', 'bleu4', 'rougeL', 'code_bleu', 'MAE']:
            if key in metrics:
                metric_value = metrics[key]
                break
        if metric_value is None and len(metrics) > 0:
            # If no priority metric is found, use the first metric
            metric_value = list(metrics.values())[0]
    else:
        metric_value = metrics

    # Simplify the file name, keeping only the last part
    model_name = model_name.split('/')[-1]
    
    if file_exists:
        # Read existing data
        rows = []
        tasks = set()
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, ['task', model_name])  # If the file is empty, use the default header
            if len(header) == 1:  # If there is only the task column, add the model column
                header.append(model_name)
            rows.append(header)

            # Read existing data and update
            for row in reader:
                if row[0] == task_name:  # If the same task is found, update the value
                    row = [task_name, str(metric_value)]
                tasks.add(row[0])
                rows.append(row)

            # If it is a new task, add a new row
            if task_name not in tasks:
                rows.append([task_name, str(metric_value)])
    else:
        # Create a new file
        rows = [
            ['task', model_name],
            [task_name, str(metric_value)]
        ]

    # Write all data
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


if __name__ == "__main__":

    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run audio understanding tasks")
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Name of the audio understanding model to use')
    parser.add_argument('-d', '--data_dir', type=str, default='./audio/understanding/', help='Directory containing task data')
    parser.add_argument('-o', '--output_dir', type=str, default='./audio/predictions/understanding/', help='Directory to save predictions')
    parser.add_argument('-r', '--root_path', type=str, default='./', help='Root path for logging performance')
    parser.add_argument('-t', '--task_names', type=str, nargs='+',
                        help='List of task names to run (default: AccentClassification AccentSexClassification AcousticSceneClassification)')
    args = parser.parse_args()

    # model_name = 'Qwen2-Audio-7B-Instruct'
    # data_dir = './understanding/'
    # output_dir = f'./predictions/understanding/{model_name}'
    # root_path = './'

    model = AudioComprehensionModel(model_name=args.model_name)


    task_name_list = [
        'AccentClassification', 'AccentSexClassification', 'AcousticSceneClassification',
        'AnimalSoundClassification', 'AudioCaptioning', 'AudioCaptioningClotho',
        'AudioQA', 'BirdSoundDetection', 'EnvironmentSoundRecognition',
        'IntentClassification', 'MusicGenreClassification',
        'MusicInstrumentClassification', 'MusicInstrumentSourceAnalysis',
        'MusicPitchAnalysis', 'NoteQualitiesAnalysis', 'OpenAQA',
        'SingerIdentification', 'SoundEventClassification',
        'SpeakerIdentification', 'SpeechCommand',
        'SpeechEmotionRecognition', 'VocalSoundClassification',
        'VocalTechniqueDetection'
    ]
    if args.task_names is None or len(args.task_names) == 0:
        args.task_names = task_name_list
    
    for task_name in args.task_names: # os.listdir(data_dir):

        # Dynamically get the class by its name
        if task_name in globals():  # Ensure the class is defined in the current scope
            task_class = globals()[task_name]
        else:
            # Optionally, handle cases where the class is not found
            print(f"Task {task_name} is not defined in the current scope.")
            continue

        # Initialize the task class
        import glob
        json_file_list = glob.glob(os.path.join(args.data_dir, task_name, "*.json"))
        if len(json_file_list) == 0:
            print(f"No JSON files found for task: {task_name}")
            continue
        elif len(json_file_list) > 1:
            print(f"Multiple JSON files found for task: {task_name}, using the first one: {json_file_list[0]}")
            task_annotation_data = json_file_list[0]
        else:
            task_annotation_data = json_file_list[0]
        task = task_class(
            task_data=task_annotation_data,
            model=model,
            audio_dir=os.path.join(args.data_dir, task_name, 'audios'),
            output_dir=args.output_dir
        )
        
        # Run inference for the task
        # This should generate audio files based on the task's data
        print(f"Running inference for task: {task_name}")
        task.run_inference()
        # if you want to save the predictions, you need to rewrite the save_predictions() in each Task class depending on your need, and call task.save_predictions() after task.run_inference() or inside the run_inference method.


        # Evaluate the task, return a dictionary of metrics
        # For example, {'FAD_score': 0.123}
        eval_results = task.evaluate()   
        print("Task name: ", task_name, "Evaluation results:", eval_results)
        log_performance_json(
            model_name=args.model_name, 
            task_name=task_name, 
            metric=list(eval_results.keys())[0].split('_')[0],   # CLAP_score
            score=eval_results[list(eval_results.keys())[0]],  # e.g., 0.123
            root_path=args.data_dir)

    # or you can run the tasks one by one like below:
    # task_name = 'AcousticSceneClassification'
    # task = AcousticSceneClassification(
    #     task_data=os.path.join(data_dir, f"{task_name}/annotation.json"),
    #     model=model,
    #     audio_dir=os.path.join(data_dir, f"{task_name}/audios"),
    #     output_dir=output_dir)
    # task.run_inference()
    # print(task.evaluate())



