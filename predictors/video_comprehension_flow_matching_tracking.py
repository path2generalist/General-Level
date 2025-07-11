import tqdm
from typing import List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
import cv2
from typing import Tuple
import os
import json
import argparse

import torch
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

def exact_match_accuracy(predictions: List[str], references: List[str]) -> float:
    correct = 0
    for pred, ref in zip(predictions, references):
        if isinstance(ref, str):
            ref = [ref]
        is_match_this_turn = False
        for r in ref:
            if pred.strip() == r.strip():
                is_match_this_turn = True
        if is_match_this_turn:
            correct += 1
    return correct / len(predictions) if predictions else 0.0


def bbox_to_corners(bbox):
    """将(x_min, y_min, w, h)格式转换为(x_min, y_min, x_max, y_max)格式"""
    x_min, y_min, w, h = bbox
    return (x_min, y_min, x_min + w, y_min + h)


def calculate_iou(bbox1, bbox2):
    """计算两个边界框的交并比(IoU/Jaccard Index)"""
    # 转换为对角坐标格式
    bbox1 = bbox_to_corners(bbox1)
    bbox2 = bbox_to_corners(bbox2)

    # 计算交集区域的坐标
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # 计算交集面积
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算两个边界框的面积
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # 计算并集面积
    union_area = bbox1_area + bbox2_area - intersection_area

    # 计算IoU
    if union_area == 0:
        return 0.0
    return intersection_area / union_area


def calculate_j_metric(pred_bboxes, gt_bboxes):
    """计算J指标(Jaccard Index)"""
    if len(pred_bboxes) != len(gt_bboxes):
        raise ValueError("预测边界框和真实边界框数量不一致")

    iou_values = []
    for pred, gt in zip(pred_bboxes, gt_bboxes):
        iou = calculate_iou(pred, gt)
        iou_values.append(iou)

    # 返回平均Jaccard Index
    return sum(iou_values) / len(iou_values) if iou_values else 0.0


def calculate_f1_score(pred_bboxes, gt_bboxes, threshold=0.5):
    """计算F1 Score(F指标)"""
    if len(pred_bboxes) == 0 and len(gt_bboxes) == 0:
        return 1.0  # 特殊情况：没有检测也没有真实目标，视为完全正确

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # 标记已匹配的真实边界框
    gt_matched = [False] * len(gt_bboxes)

    # 计算每对边界框的IoU
    iou_matrix = []
    for i, pred in enumerate(pred_bboxes):
        row = []
        for j, gt in enumerate(gt_bboxes):
            row.append(calculate_iou(pred, gt))
        iou_matrix.append(row)

    # 贪心匹配：将每个预测边界框匹配到IoU最高的真实边界框
    for i in range(len(pred_bboxes)):
        if not iou_matrix:
            break

        # 找到当前行的最大值及其索引
        max_iou = max(iou_matrix[i]) if iou_matrix[i] else 0
        j = iou_matrix[i].index(max_iou) if iou_matrix[i] else -1

        if max_iou >= threshold:
            true_positives += 1
            gt_matched[j] = True
        else:
            false_positives += 1

    # 计算假阴性
    false_negatives = sum(1 for matched in gt_matched if not matched)

    # 计算精确率和召回率
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # 计算F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def calculate_j_and_f_metrics(pred_bboxes, gt_bboxes, iou_threshold=0.5):
    """计算J指标和F指标"""
    # 计算J指标
    j_metric = calculate_j_metric(pred_bboxes, gt_bboxes)

    # 计算F指标
    f_metric = calculate_f1_score(pred_bboxes, gt_bboxes, threshold=iou_threshold)

    return {
        "J_metric": j_metric,
        "F_metric": f_metric
    }

def read_flow(file_path: str) -> np.ndarray:
    if file_path.endswith('.flo'):
        return read_flow_flo(file_path)
    elif file_path.endswith(('.png', '.jpg', '.jpeg')):
        return read_flow_png(file_path)
    else:
        raise NotImplementedError


def read_flow_flo(file_path: str) -> np.ndarray:
    with open(file_path, 'rb') as f:

        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            raise NotImplementedError

        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]

        flow = np.fromfile(f, np.float32, count=2 * w * h)
        flow = flow.reshape(h, w, 2)

    return flow


def read_flow_png(file_path: str) -> np.ndarray:
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    # 确保图像有足够的通道
    if len(img.shape) != 3 or img.shape[2] < 2:
        raise NotImplementedError

    u = (img[:, :, 2] - 32768.0) / 64.0  # R
    v = (img[:, :, 1] - 32768.0) / 64.0  # G

    flow = np.stack([u, v], axis=2)

    return flow


def calculate_epe(flow_gt: np.ndarray, flow_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    if flow_gt.shape != flow_pred.shape:
        raise NotImplementedError

    diff = flow_gt - flow_pred
    epe_map = np.sqrt(np.sum(diff ** 2, axis=2))

    mean_epe = np.mean(epe_map)

    return mean_epe, epe_map

class Sa2VAModel:
    def __init__(self, model_name="ByteDance/Sa2VA-4B"):
        self.model_name = model_name

        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval().cuda()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.model = model
        self.tokenizer = tokenizer

    def generate(self, input_dict):
        pred_dict = self.model.predict_forward(**input_dict, tokenizer=self.tokenizer)
        if 'prediction_masks' in pred_dict.keys() and pred_dict['prediction_masks'] and len(
                pred_dict['prediction_masks']) != 0:
            masks = pred_dict['prediction_masks'][0]  # (f, h, w)
        else:
            masks = None
        text_response = pred_dict["prediction"]
        return text_response, masks

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

def get_bbox_from_mask(mask):
    if len(mask.shape) != 2:
        raise NotImplementedError

    y_indices, x_indices = np.nonzero(mask)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    x_min = np.min(x_indices)
    x_max = np.max(x_indices)
    y_min = np.min(y_indices)
    y_max = np.max(y_indices)

    return (x_min, y_min, x_max-x_min, y_max-y_min)

def mask2bbox(masks, video_length):
    if masks is None:
        bboxes = [[0, 0, 0, 0]] * video_length
    else:
        bboxes = []
        for mask in masks:
            bbox = get_bbox_from_mask(mask)
            if bbox is None:
                bbox = [0, 0, 0, 0]
            bboxes.append(bbox)
    return bboxes

class MatchTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"])
                for d in task_data["data"]]

    def run_inference(self):
        self.predictions = []
        self.references = []
        for inst in tqdm.tqdm(self.data):
            prompt = "<image>\n" + inst.input["prompt"]
            video_folder = inst.input["video_folder"]
            frame_files = [os.path.join(video_folder, _name) for _name in os.listdir(video_folder)]
            video = []
            for image_path in frame_files:
                video.append(Image.open(image_path).convert('RGB'))

            input_dict = {
                "video": video,
                "text": prompt,
            }

            response, _ = self.model.generate(input_dict, max_new_tokens=256)
            response = response.split("<")[0].strip()

            self.predictions.append(response)
            self.references.append(inst.output["answer"])

    def evaluate(self) -> Dict[str, float]:
        acc = exact_match_accuracy(self.predictions, self.references)
        return {"accuracy": acc}

class TrackingTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"])
                for d in task_data["data"]]

    def run_inference(self):
        self.predictions = []
        self.references = []
        for inst in tqdm.tqdm(self.data):
            prompt = "<image>\n" + inst.input["prompt"]
            video_folder = inst.input["video_folder"]
            frame_files = [os.path.join(video_folder, _name) for _name in os.listdir(video_folder)]
            video = []
            for image_path in frame_files:
                video.append(Image.open(image_path).convert('RGB'))

            input_dict = {
                "video": video,
                "text": prompt,
            }

            response, masks = self.model.generate(input_dict, max_new_tokens=256)

            bboxes = mask2bbox(masks, len(video))

            self.predictions.append(bboxes)
            self.references.append(inst.output["answer"])

    def evaluate(self) -> Dict[str, float]:
        j_f, n = 0, 1e-4
        for pred_bboxes, gt_bboxes in zip(self.predictions, self.references):
            metrics = calculate_j_and_f_metrics(pred_bboxes, gt_bboxes)
            j_f += (metrics['J_metric'] + metrics['F_metric']) / 2.0
            n += 1
        j_f = j_f / n
        return {"J&F": j_f}

class FlowTask(BaseTask):
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"])
                for d in task_data["data"]]

    def run_inference(self):
        self.predictions = []
        self.references = []
        for inst in tqdm.tqdm(self.data):
            prompt = "<image>\n" + inst.input["prompt"]
            video_folder = inst.input["video_folder"]
            frame_files = [os.path.join(video_folder, _name) for _name in os.listdir(video_folder)]
            video = []
            for image_path in frame_files:
                video.append(Image.open(image_path).convert('RGB'))

            input_dict = {
                "video": video,
                "text": prompt,
            }

            response, masks = self.model.generate(input_dict, max_new_tokens=256)

            pred_flows = np.zeros(masks.shape[1], masks.shape[2], 2)

            self.predictions.append(pred_flows)
            self.references.append(read_flow(inst.output["flow"]))

    def evaluate(self) -> Dict[str, float]:
        EPE, n = 0, 1e-4
        for pred_flow, gt_flow in zip(self.predictions, self.references):
            mean_epe, _ = calculate_epe(pred_flow, gt_flow)
            EPE += mean_epe
            n += 1
        EPE = EPE / n
        return {"EPE": EPE}


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


def log_performance_detail(model_name, task_name, metrics, root_path, output_file='performance_log.csv'):
    import csv
    file_path = os.path.join(root_path, output_file)
    file_exists = os.path.isfile(file_path)

    # 从metrics字典中获取主要指标值
    metric_value = None
    if isinstance(metrics, dict):
        # 按照优先级选择指标
        for key in ['accuracy', 'f1', 'micro_f1', 'bleu4', 'rougeL', 'code_bleu', 'MAE']:
            if key in metrics:
                metric_value = metrics[key]
                break
        if metric_value is None and len(metrics) > 0:
            # 如果没有找到优先指标，使用第一个指标
            metric_value = list(metrics.values())[0]
    else:
        metric_value = metrics

    # 简化文件名，只保留最后一部分
    model_name = model_name.split('/')[-1]

    if file_exists:
        # 读取现有数据
        rows = []
        tasks = set()
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, ['task', model_name])  # 如果文件为空，使用默认表头
            if len(header) == 1:  # 如果只有task列，添加model列
                header.append(model_name)
            rows.append(header)

            # 读取现有数据并更新
            for row in reader:
                if row[0] == task_name:  # 如果找到相同任务，更新值
                    row = [task_name, str(metric_value)]
                tasks.add(row[0])
                rows.append(row)

            # 如果是新任务，添加新行
            if task_name not in tasks:
                rows.append([task_name, str(metric_value)])
    else:
        # 创建新文件
        rows = [
            ['task', model_name],
            [task_name, str(metric_value)]
        ]

    # 写入所有数据
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="General-Bench-Openset/video/comprehension")
    parser.add_argument("--model_name", type=str, default="ByteDance/Sa2VA-4B")
    args = parser.parse_args()
    root_path = args.root_path
    model_name = args.model_name

    model = Sa2VAModel(model_name=model_name)

    task_files = [
        "AnimalTrack",
        "GreenWaterTrack",
        "LongVideoHumanTrack",
        "RelationMatch",
        "UAVUAVTrack",
        "BallTrack",
        "HumanPartTrack",
        "LongVideoVehicleTrack",
        "ShapeMatch",
        "UAVVehicleTrack",
        "BlueWaterTrack",
        "HumanTrack",
        "MotionMatch",
        "SizeMatch",
        "VehicleTrack",
        "ColorMatch",
        "LOGOMarkerMatch",
        "ObjectMarkerMatch",
        "SyntheticSceneFlowEstimate",
        "WhiteWaterTrack",
        "ComplexSceneFlowEstimate",
        "LongVideoAnimalTrack",
        "OtherPartTrack",
        "UAVBuildingTrack",
        "YellowWaterTrack",
        "CrowdTrack",
        "LongVideoCrowdTrack",
        "PanoramicFlowEstimate",
        "UAVGeneralObjectTrack",
        "GeneralObjectTrack",
        "LongVideoGeneralObjectTrack",
        "PositionMatch",
        "UAVHumanTrack"]

    task_files = [w + '.json' if not w.endswith('json') else w for w in task_files]

    if isinstance(task_files, str):
        task_files = [task_files]

    for idx, filename in enumerate(task_files):
        file_path = os.path.join(root_path, f"{filename.replace('.json', '')}/", filename)
        if not os.path.exists(file_path):
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            task_data = json.load(f)

        task_type = task_data["type"]
        task_name = task_data["task"]
        print(f"Running evaluation for task {idx + 1}: {task_name}")

        # 定义任务类型与任务类的映射字典
        TASK_MAPPING = {
            "AnimalTrack": TrackingTask,
            "GreenWaterTrack": TrackingTask,
            "LongVideoHumanTrack": TrackingTask,
            "RelationMatch": MatchTask,
            "UAVUAVTrack": TrackingTask,
            "BallTrack": TrackingTask,
            "HumanPartTrack": TrackingTask,
            "LongVideoVehicleTrack": TrackingTask,
            "ShapeMatch": MatchTask,
            "UAVVehicleTrack": TrackingTask,
            "BlueWaterTrack": TrackingTask,
            "HumanTrack": TrackingTask,
            "MotionMatch": MatchTask,
            "SizeMatch": MatchTask,
            "VehicleTrack": TrackingTask,
            "ColorMatch": MatchTask,
            "LOGOMarkerMatch": MatchTask,
            "ObjectMarkerMatch": MatchTask,
            "SyntheticSceneFlowEstimate": FlowTask,
            "WhiteWaterTrack": TrackingTask,
            "ComplexSceneFlowEstimate": FlowTask,
            "LongVideoAnimalTrack": TrackingTask,
            "OtherPartTrack": TrackingTask,
            "UAVBuildingTrack": TrackingTask,
            "YellowWaterTrack": TrackingTask,
            "CrowdTrack": TrackingTask,
            "LongVideoCrowdTrack": TrackingTask,
            "PanoramicFlowEstimate": FlowTask,
            "UAVGeneralObjectTrack": TrackingTask,
            "GeneralObjectTrack": TrackingTask,
            "LongVideoGeneralObjectTrack": TrackingTask,
            "PositionMatch": MatchTask,
            "UAVHumanTrack": TrackingTask,
        }

        # 根据 task_type 获取对应的任务类
        task_class = TASK_MAPPING.get(task_type)  # 使用精确匹配
        if task_class is None:
            raise NotImplementedError
        else:
            task = task_class(task_data, model)

        task.run_inference()
        metrics = task.evaluate()
        print("Task name: ", task_name, "Task type: ", task_type, "Evaluation results:", metrics)
        log_performance(model_name, task_name, metrics, root_path)