import tqdm
from typing import List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
import os
import json
import argparse

import torch
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          LlavaOnevisionForConditionalGeneration, AutoProcessor)

# An example of the model
class LLavaOneVisionModel:
    def __init__(self, model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf"):
        self.model_name = model_name

        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).eval().cuda()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.model = model
        self.tokenizer = tokenizer

    def generate(self, conversation, video):
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=video, text=prompt, return_tensors="pt").to(self.model.device, torch.float16)
        outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        text_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        text_response = text_response.split('assistant\n')[1]

        return text_response

@dataclass
class Instance:
    input: Dict[str, Any]
    output: Dict[str, Any]
    id: str


class BaseTask(ABC):
    def __init__(self, task_data: Dict[str, Any], model):
        self.task_data = task_data
        self.model = model
        self.data = self._parse_data(task_data)

    @abstractmethod
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def run_inference(self):
        pass


def cal_accuracy(predictions: List[str], references: List[str]) -> float:
    correct = 0
    for pred, ref in zip(predictions, references):
        if isinstance(ref, str):
            ref = [ref]
        is_match_this_turn = False
        for r in ref:
            if "yes" in r.lower() or "no" in r.lower():
                # for yes or no question
                r = r.lower()
                pred = pred.lower()

            if r.strip() in pred.strip():
                is_match_this_turn = True
            
        if is_match_this_turn:
            correct += 1
    return correct / len(predictions) if predictions else 0.0


class Bleu1_Scorer():
    def __init__(self, predictions, references):
        from pycocoevalcap.bleu.bleu import Bleu
        self.pred = predictions
        self.gt = references
        self.scorers = [
            (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('Computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.pred)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    print('%s: %0.3f' % (m, sc * 100))
                total_scores['Bleu'] = [x * 100 for x in score]
            else:
                total_scores[method] = score * 100
        
        return {"Bleu_1": total_scores['Bleu'][0]}
    

class AccTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        self.task_name = task_data["task"]
        return [Instance(input=d["input"], output=d["output"], id=d["id"])
                for d in task_data["data"]]

    def read_video_frames(self, data_path_list, root_path, max_frames_num=64):
        frames = []
        if len(data_path_list) > max_frames_num:
            frame_idx = np.linspace(0, len(data_path_list) - 1, max_frames_num, dtype=int)
            data_path_list = [data_path_list[i] for i in frame_idx]

        for frame_path in data_path_list:
            path = os.path.join(root_path, frame_path)
            if os.path.exists(path):
                try:
                    frame = Image.open(path)
                    frames.append(frame)
                except Exception as e:
                    print(f"Warning: Failed to read frame {path}. Error: {e}")
            else:
                print(f"Warning: Frame path {path} does not exist.")
        return frames


    def run_inference(self, root_path):

        if os.path.exists(f'./predictions_{self.task_name}.json'):
            self.predictions = json.load(open(f'./predictions_{self.task_name}.json', 'r'))
            self.references = json.load(open(f'./references_{self.task_name}.json', 'r'))
            return
        
        self.predictions = []
        self.references = []
        for inst in tqdm.tqdm(self.data):
            video_path = inst.input['video_file_list']
            video = self.read_video_frames(video_path, os.path.join(root_path, self.task_name, 'videos'), max_frames_num=64)

            question = 'Please answer the following question related to the video. ' + inst.input['prompt']

            other_requirements = ''
            if 'VideoActionCounting' in self.task_name:
                other_requirements = 'The output must consist only of Arabic numerals.'
            if 'VideoActionOrdering' in self.task_name:
                other_requirements = 'The output format must be: [num]->[num]->[num]->[num]. The number represents the index marked in the question. For example: 2->1->3->4, 1->2->3->4, 3->2->1->4...'
            if 'SignLanguageVideoRecognition' in self.task_name:
                other_requirements = 'The output format must be a word.'
            question += other_requirements

            conversation = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "video"},
                    ],
                },
            ]

            text_response = self.model.generate(conversation, video)

            self.predictions.append(text_response)
            self.references.append(inst.output["text"])
        
        json.dump(self.predictions, open(f'./predictions_{self.task_name}.json', 'w'))
        json.dump(self.references, open(f'./references_{self.task_name}.json', 'w'))

    def evaluate(self) -> Dict[str, float]:

        acc = cal_accuracy(self.predictions, self.references)
        return {"accuracy": acc*100}
    

class BLEUTASK(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        self.task_name = task_data["task"]
        return [Instance(input=d["input"], output=d["output"], id=d["id"])
                for d in task_data["data"]]

    def read_video_frames(self, data_path_list, root_path, max_frames_num=64):
        frames = []
        if len(data_path_list) > max_frames_num:
            frame_idx = np.linspace(0, len(data_path_list) - 1, max_frames_num, dtype=int)
            data_path_list = [data_path_list[i] for i in frame_idx]

        for frame_path in data_path_list:
            path = os.path.join(root_path, frame_path)
            if os.path.exists(path):
                try:
                    frame = Image.open(path)
                    frames.append(frame)
                except Exception as e:
                    print(f"Warning: Failed to read frame {path}. Error: {e}")
            else:
                print(f"Warning: Frame path {path} does not exist.")
        return frames


    def run_inference(self, root_path):

        if os.path.exists(f'./predictions_{self.task_name}.json'):
            self.predictions = json.load(open(f'./predictions_{self.task_name}.json', 'r'))
            self.references = json.load(open(f'./references_{self.task_name}.json', 'r'))
            return
        
        self.predictions = []
        self.references = []
        for inst in tqdm.tqdm(self.data):
            video_path = inst.input['video_file_list']
            video = self.read_video_frames(video_path, os.path.join(root_path, self.task_name, 'videos'), max_frames_num=64)

            question = 'Please answer the following question related to the video. ' + inst.input['prompt']
            other_requirements = ' The output should be concise. '
            question += other_requirements

            conversation = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "video"},
                    ],
                },
            ]

            text_response = self.model.generate(conversation, video)

            self.predictions.append(text_response)
            self.references.append(inst.output["text"])
        
        json.dump(self.predictions, open(f'./predictions_{self.task_name}.json', 'w'))
        json.dump(self.references, open(f'./references_{self.task_name}.json', 'w'))

    def evaluate(self) -> Dict[str, float]:

        predictions = {}
        references = {}

        num = 1
        for pred, ref in zip(self.predictions, self.references):
            predictions[str(num)] = [pred.lower()]
            references[str(num)] = [ref.lower()]
            num += 1

        bleu1_scorer = Bleu1_Scorer(predictions, references)
        bleu1_scores = bleu1_scorer.compute_scores()
        return bleu1_scores



def log_performance(model_name, task_name, metrics, root_path, output_file='performance_log.csv'):
    import csv
    file_exists = os.path.isfile(os.path.join(root_path, output_file))

    row_data = {
        'model': model_name,
        'task': task_name,
        'metrics': str(metrics)
    }

    with open(os.path.join(root_path, output_file), mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()

        writer.writerow(row_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="General-Bench-Openset/video/comprehension")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    args = parser.parse_args()
    root_path = args.root_path
    model_name = args.model_name

    model = LLavaOneVisionModel(model_name=model_name) # An example of the model

    # 56 tasks
    task_files = [
        "AgricultureVideoQuestionAnswering",
        "ArtRecognition",
        "ArtsAndCraftsVideoCaptioning",
        "AutosAndVehiclesVideoCaptioning",
        "BallGameVideoQuestionAnswering",
        "BallSportsVideoCaptioning",
        "BodyMotionVideoCaptioning",
        "BusinessVideoCaptioning",
        "ComedyVideoQuestionAnswering",
        "DailyLifeAndSkillsVideoCaptioning",
        "EducationVideoQuestionAnswering",
        "EntertainmentRelatedVideoCaptioning",
        "FacialActionVideoCaptioning",
        "FacialObjectOperationsVideoCaptioning",
        "FinanceVideoCaptioning",
        "FoodVideoCaptioning",
        "GameVideoQuestionAnswering",
        "GeographyVideoQuestionAnswering",
        "GymnasticsVideoQuestionAnswering",
        "HistoryAndLiteratureVideoCaptioning",
        "HumanHumanInteractionVideoCaptioning",
        "HumanObjectInteractionVideoCaptioning",
        "HumanObjectInteractionVideoQuestionAnswering",
        "HumanSurvivalVideoQuestionAnswering",
        "HumorVideoCaptioning",
        "MilitaryVideoQuestionAnswering",
        "MovieAndShowVideoCaptioning",
        "MovieVideoQuestionAnswering",
        "MusicalInstrumentsVideoCaptioning",
        "MusicVideoQuestionAnswering",
        "NaturalDisasterVideoRecognition",
        "NewsAndDocumentaryVideoCaptioning",
        "ObjectColorVideoQuestionAnswering",
        "ObjectDirectionVideoQuestionAnswering",
        "ObjectLocationVideoQuestionAnswering",
        "ObjectMotionVideoQuestionAnswering",
        "PersonalCareVideoCaptioning",
        "PetsVideoQuestionAnswering",
        "PetsVideoRecognition",
        "ScienceAndTechnologyVideoCaptioning",
        "ScienceVideoQuestionAnswering",
        "ScienceVideoRecognition",
        "SignLanguageVideoRecognition",
        "SportsAndExcerciseVideoCaptioning",
        "SportsVideoQuestionAnswering",
        "TVShowRecognition",
        "VideoActionCounting",
        "VideoActionOrdering",
        "VideoActionSequencePrediction",
        "VideoActionSequenceUnderstanding",
        "VideoAnimalRecognition",
        "VideoFoodRecognition",
        "VideoObjectCounting",
        "VideoObjectExistenceRecognition",
        "VideoObjectInteractionRecognition",
        "VideoSportsRecognition",
    ]

    task_files = [w + '.json' if not w.endswith('json') else w for w in task_files]

    if isinstance(task_files, str):
        task_files = [task_files]

    for idx, filename in enumerate(task_files):
        file_path = os.path.join(root_path, f"{filename.replace('.json', '')}/", "annotation.json")

        if not os.path.exists(file_path):
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            task_data = json.load(f)

        task_type = task_data["type"]
        task_name = task_data["task"]
        print(f"Running evaluation for task {idx + 1}: {task_name}")

        TASK_MAPPING = {
            "AgricultureVideoQuestionAnswering": BLEUTASK,
            "ArtRecognition": AccTask,
            "ArtsAndCraftsVideoCaptioning": BLEUTASK,
            "AutosAndVehiclesVideoCaptioning": BLEUTASK,
            "BallGameVideoQuestionAnswering": AccTask,
            "BallSportsVideoCaptioning": BLEUTASK,
            "BodyMotionVideoCaptioning": BLEUTASK,
            "BusinessVideoCaptioning": BLEUTASK,
            "ComedyVideoQuestionAnswering": BLEUTASK,
            "DailyLifeAndSkillsVideoCaptioning": BLEUTASK,
            "EducationVideoQuestionAnswering": AccTask,
            "EntertainmentRelatedVideoCaptioning": BLEUTASK,
            "FacialActionVideoCaptioning": BLEUTASK,
            "FacialObjectOperationsVideoCaptioning": BLEUTASK,
            "FinanceVideoCaptioning": BLEUTASK,
            "FoodVideoCaptioning": BLEUTASK,
            "GameVideoQuestionAnswering": BLEUTASK,
            "GeographyVideoQuestionAnswering": BLEUTASK,
            "GymnasticsVideoQuestionAnswering": AccTask,
            "HistoryAndLiteratureVideoCaptioning": BLEUTASK,
            "HumanHumanInteractionVideoCaptioning": BLEUTASK,
            "HumanObjectInteractionVideoCaptioning": BLEUTASK,
            "HumanObjectInteractionVideoQuestionAnswering": BLEUTASK,
            "HumanSurvivalVideoQuestionAnswering": BLEUTASK,
            "HumorVideoCaptioning": BLEUTASK,
            "MilitaryVideoQuestionAnswering": BLEUTASK,
            "MovieAndShowVideoCaptioning": BLEUTASK,
            "MovieVideoQuestionAnswering": BLEUTASK,
            "MusicalInstrumentsVideoCaptioning": BLEUTASK,
            "MusicVideoQuestionAnswering": BLEUTASK,
            "NaturalDisasterVideoRecognition": BLEUTASK,
            "NewsAndDocumentaryVideoCaptioning": BLEUTASK,
            "ObjectColorVideoQuestionAnswering": AccTask,
            "ObjectDirectionVideoQuestionAnswering": BLEUTASK,
            "ObjectLocationVideoQuestionAnswering": AccTask,
            "ObjectMotionVideoQuestionAnswering": AccTask,
            "PersonalCareVideoCaptioning": BLEUTASK,
            "PetsVideoQuestionAnswering": BLEUTASK,
            "PetsVideoRecognition": BLEUTASK,
            "ScienceAndTechnologyVideoCaptioning": BLEUTASK,
            "ScienceVideoQuestionAnswering": BLEUTASK,
            "ScienceVideoRecognition": BLEUTASK,
            "SignLanguageVideoRecognition": AccTask,
            "SportsAndExcerciseVideoCaptioning": BLEUTASK,
            "SportsVideoQuestionAnswering": BLEUTASK,
            "TVShowRecognition": AccTask,
            "VideoActionCounting": AccTask,
            "VideoActionOrdering": AccTask,
            "VideoActionSequencePrediction": BLEUTASK,
            "VideoActionSequenceUnderstanding": BLEUTASK,
            "VideoAnimalRecognition": AccTask,
            "VideoFoodRecognition": AccTask,
            "VideoObjectCounting": BLEUTASK,
            "VideoObjectExistenceRecognition": BLEUTASK,
            "VideoObjectInteractionRecognition": BLEUTASK,
            "VideoSportsRecognition": AccTask,
        }

        task_class = TASK_MAPPING.get(task_name)
        if task_class is None:
            raise NotImplementedError
        else:
            task = task_class(task_data, model)

        task.run_inference(root_path=root_path)
        metrics = task.evaluate()

        print("Task name: ", task_name, "Task type: ", task_type, "Evaluation results:", metrics)
        log_performance(model_name, task_name, metrics, '../outcome/', output_file='video_comprehension_qa_caption_performance_log.csv')



