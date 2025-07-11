from typing import List
from utils.data_types import ModalityType, TaskType, TaskResult
from utils.base_processor import BaseModalityProcessor

class ImageProcessor(BaseModalityProcessor):
    """图像模态处理器"""
    def __init__(self, modality: ModalityType, dataset_dir: str, pred_json_file: str):
        super().__init__(modality, dataset_dir, pred_json_file)
    
    def process_1(self):
        return []
    
    def process_comprehension(self) -> List[TaskResult]:
        """处理图像理解类任务
        
        需要返回一个TaskResult列表，每个TaskResult包含：
        - task_name: 任务名称，例如 "image_classification", "object_detection" 等
        - metric: 评估指标，例如 "accuracy", "mAP" 等
        - score: 评估分数
        - task_type: 默认为 TaskType.COMPREHENSION，不需要指定
        
        示例格式：
        return [
            TaskResult(
                task_name="image_classification",
                metric="accuracy",
                score=0.95
            ),
            TaskResult(
                task_name="object_detection",
                metric="mAP",
                score=0.82
            )
        ]
        """
        return []
    
    def process_generation(self) -> List[TaskResult]:
        """处理图像生成类任务
        
        需要返回一个TaskResult列表，每个TaskResult包含：
        - task_name: 任务名称，例如 "image_generation", "image_editing" 等
        - metric: 评估指标，例如 "FID", "IS" 等
        - score: 评估分数
        - task_type: 需要指定为 TaskType.GENERATION
        
        示例格式：
        return [
            TaskResult(
                task_name="image_generation",
                metric="FID",
                score=15.2,
                task_type=TaskType.GENERATION
            ),
            TaskResult(
                task_name="image_editing",
                metric="PSNR",
                score=28.5,
                task_type=TaskType.GENERATION
            )
        ]
        """
        return []

# 使用示例
if __name__ == "__main__":
    processor = ImageProcessor(ModalityType.IMAGE, "")
    
    # 测试理解任务
    print("\n理解类任务结果:")
    for task in processor.process_comprehension():
        print(f"任务: {task.task_name}")
        print(f"指标: {task.metric}")
        print(f"分数: {task.score}")
        print("-" * 20)
    
    # 测试生成任务
    print("\n生成类任务结果:")
    for task in processor.process_generation():
        print(f"任务: {task.task_name}")
        print(f"指标: {task.metric}")
        print(f"分数: {task.score}")
        print("-" * 20) 