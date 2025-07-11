import subprocess
from typing import List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import clip
import torch
from torch import nn
import torch.nn.functional as F

from typing import Tuple
import os
import json
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from video_generation_evaluation.toolkit.fvd import get_dataset_features, I3DFeatureExtractor
from numpy import cov
from numpy import mean
from scipy.linalg import sqrtm
from video_generation_evaluation.evaluate import task2dimension


class BaseTask(ABC):
    def __init__(self, task_data: str, model):
        self.task_data = task_data
        self.model = model
        self.data = self._parse_data(task_data)

    @abstractmethod
    def _parse_data(self, task_data: Dict[str, Any]):
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def run_inference(self):
        pass

class T2VTask(BaseTask):
    def _parse_result_file(self, output_dir: Path) -> float | None:
        for jsonfile in output_dir.iterdir():
            if "eval" in jsonfile.name:
                with open(jsonfile.as_posix(), "r") as file:
                    data = json.load(file)
            
        return float(data[self.taskname][0])
    
    def _parse_data(self, task_data):
        with open(task_data, "r") as file:
            annos = json.load(file)
        taskname = annos["task"].replace(" ", "")
        self.taskname = taskname
        self.save_root = os.path.join("General-Bench", "Video-Generation", taskname)
        return annos["data"]

    def run_inference(self):
        for d in self.data:
            prompt = d["input"]["prompt"]
            for i in range(5):
                video = self.model(prompt, generator=torch.Generator(self.model.device).manual_seed(i)).frames[0]
                save_name = prompt + "-" + str(i) + ".mp4"
                save_path = os.path.join(self.save_root, save_name)
                export_to_video(video, save_path, fps=8)

class FVDEval(T2VTask):
    def evaluate(self, real_video_root):
        model = I3DFeatureExtractor().cuda().eval()
        
        real_features = get_dataset_features(real_video_root, model)
        generated_features = get_dataset_features(self.save_root, model)

        mu_real = mean(real_features, axis=0)
        mu_generated = mean(generated_features, axis=0)

        sigma_real = cov(real_features, rowvar=False)
        sigma_generated = cov(generated_features, rowvar=False)

        diff = mu_real - mu_generated
        covmean, _ = sqrtm(sigma_real.dot(sigma_generated), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fvd = diff.dot(diff) + np.trace(sigma_real + sigma_generated - 2 * covmean)
        print(f"{self.taskname} score: {fvd}")
        return fvd

class ThirdPartyEval(T2VTask):
    def evaluate(self):
        videos_path = Path(self.save_root).resolve()
        dimension    = task2dimension[self.taskname]
        full_info    = Path("./full_info_t2v.json").resolve()
        output_dir   = Path("./evaluation_results").resolve()
        output_dir   = output_dir.joinpath(self.taskname)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "python", "-W", "ignore", "evaluate.py",
            "--full_json_dir", str(full_info),
            "--videos_path",  str(videos_path),
            "--dimension",    dimension,
            "--output_path",  str(output_dir)
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Evaluation failed: {exc}") from exc
    
        score = self._parse_result_file(Path(output_dir))
        print(f"{self.taskname} score: {score}")
        return score

class I2VTask(BaseTask):
    def _parse_result_file(self, output_dir: Path) -> float | None:
        score = 0
        for jsonfile in output_dir.iterdir():
            if "eval" in jsonfile.name:
                with open(jsonfile.as_posix(), "r") as file:
                    data: dict = json.load(file)
                score += list(data.values())[0][0]
        return score
                    
    def _parse_data(self, task_data):
        self.dirpath = os.path.dirname(task_data)
        with open(task_data, "r") as file:
            annos = json.load(file)
        taskname = annos["task"].replace(" ", "")
        self.taskname = taskname
        self.dimensions = ("subject_consistency", "overall_consistency", "motion_smoothness", "dynamic_degree")
        self.save_root = os.path.join("General-Bench", "Video-Generation", taskname)

    def run_inference(self):
        for d in self.data:
            prompt = d["input"]["prompt"]
            image = d["input"]["image"]
            image = os.path.join(self.dirpath, image)
            for i in range(5):
                video = self.model(
                    prompt=prompt,
                    image=image, 
                    generator=torch.Generator(self.model.device).manual_seed(i)
                ).frames[0]
                save_name = prompt + "-" + str(i) + ".mp4"
                save_path = os.path.join(self.save_root, save_name)
                export_to_video(video, save_path, fps=8)
    
    def evaluate(self):
        taskname = self.taskname
        full_info  = Path("./full_info_i2v.json").resolve()
        output_dir = Path("./evaluation_results").resolve()
        output_dir = output_dir.joinpath(taskname)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for dimension in self.dimensions:
            cmd = [
                "python", "-W", "ignore", "evaluate.py",
                "--full_json_dir", str(full_info),
                "--videos_path",  str(self.save_root),
                "--dimension",    dimension,
                "--output_path",  str(output_dir)
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"Evaluation failed: {exc}") from exc

        score = self._parse_result_file(Path(output_dir))
        print(f"{self.taskname} score: {score}")
        return score   
        
class AthleticsT2V(FVDEval): pass

class HumanT2V(FVDEval): pass

class ConcertT2V(FVDEval): pass

class TerrestrialAnimalT2V(FVDEval): pass

class WaterSportsT2V(FVDEval): pass

class ActionT2V(ThirdPartyEval): pass

class ArtisticT2V(ThirdPartyEval): pass

class BackgroundConsistency(ThirdPartyEval): pass

class CameraMotionT2V(ThirdPartyEval): pass

class ClassConditionedT2V(ThirdPartyEval): pass

class ColorT2V(ThirdPartyEval): pass

class DynamicT2V(ThirdPartyEval): pass

class MaterialT2V(ThirdPartyEval): pass

class MultiClassConditionedT2V(ThirdPartyEval): pass

class SceneT2V(ThirdPartyEval): pass

class SpatialRelationT2V(ThirdPartyEval): pass

class StaticT2V(ThirdPartyEval): pass

class StyleT2V(ThirdPartyEval): pass

class ArchitectureI2V(I2VTask): pass

class ClothI2V(I2VTask): pass

class FoodI2V(I2VTask): pass

class FurnitureI2V(I2VTask): pass

class HumanI2V(I2VTask): pass

class PetI2V(I2VTask): pass

class PlantI2V(I2VTask): pass

class SceneI2V(I2VTask): pass

class VehicleI2V(I2VTask): pass

class WeatherI2V(I2VTask): pass

class WildAnimalI2V(I2VTask): pass


if __name__ == "__main__":
    root = Path("General-Bench-Openset/video/generation")
    
    task_type = "T2V"
    model = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.bfloat16).to("cuda")
    
    task_files = [
        "AthleticsT2V",
        "HumanT2V",
        "ConcertT2V",
        "TerrestrialAnimalT2V",
        "WaterSportsT2V",
        "ActionT2V",
        "ArtisticT2V",
        "BackgroundConsistency",
        "CameraMotionT2V",
        "ClassConditionedT2V",
        "ColorT2V",
        "DynamicT2V",
        "MaterialT2V",
        "MultiClassConditionedT2V",
        "SceneT2V",
        "SpatialRelationT2V",
        "StaticT2V",
        "StyleT2V",
        "ArchitectureI2V",
        "ClothI2V",
        "FoodI2V",
        "FurnitureI2V",
        "HumanI2V",
        "PetI2V",
        "PlantI2V",
        "SceneI2V",
        "VehicleI2V",
        "WeatherI2V",
        "WildAnimalI2V",
    ]

    task_files = [root.joinpath(task, "annotation.json") for task in task_files]

    for idx, file in enumerate(task_files):
        if file.exists():
            continue

        with open(file.as_posix(), 'r', encoding='utf-8') as f:
            task_data = json.load(f)

        task_name = task_data["task"]
        print(f"Running evaluation for task {idx + 1}: {task_name}")

        TASK_MAPPING = {
            "AthleticsT2V":               AthleticsT2V,
            "HumanT2V":                   HumanT2V,
            "ConcertT2V":                 ConcertT2V,
            "TerrestrialAnimalT2V":       TerrestrialAnimalT2V,
            "WaterSportsT2V":             WaterSportsT2V,
            "ActionT2V":                  ActionT2V,
            "ArtisticT2V":                ArtisticT2V,
            "BackgroundConsistency":      BackgroundConsistency,
            "CameraMotionT2V":            CameraMotionT2V,
            "ClassConditionedT2V":        ClassConditionedT2V,
            "ColorT2V":                   ColorT2V,
            "DynamicT2V":                 DynamicT2V,
            "MaterialT2V":                MaterialT2V,
            "MultiClassConditionedT2V":   MultiClassConditionedT2V,
            "SceneT2V":                   SceneT2V,
            "SpatialRelationT2V":         SpatialRelationT2V,
            "StaticT2V":                  StaticT2V,
            "StyleT2V":                   StyleT2V,
            "ArchitectureI2V":            ArchitectureI2V,
            "ClothI2V":                   ClothI2V,
            "FoodI2V":                    FoodI2V,
            "FurnitureI2V":               FurnitureI2V,
            "HumanI2V":                   HumanI2V,
            "PetI2V":                     PetI2V,
            "PlantI2V":                   PlantI2V,
            "SceneI2V":                   SceneI2V,
            "VehicleI2V":                 VehicleI2V,
            "WeatherI2V":                 WeatherI2V,
            "WildAnimalI2V":              WildAnimalI2V,
        }

        clean_task_name = task_name.replace(" ", "")
        task_class = TASK_MAPPING.get(clean_task_name)
        if task_class is None:
            raise NotImplementedError
        elif task_type not in clean_task_name:
            continue
        else:
            task = task_class(file.as_posix(), model)

        task.run_inference()
        metrics = task.evaluate()
        print("Task name: ", task_name, "Task type: ", task_type, "Evaluation results:", metrics)