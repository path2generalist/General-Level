from typing import List
from utils.data_types import ModalityType, TaskType, TaskResult
from utils.base_processor import BaseModalityProcessor

class VideoProcessor(BaseModalityProcessor):
    """视频模态处理器"""
    def __init__(self, modality: ModalityType, dataset_dir: str, pred_json_file: str):
        super().__init__(modality, dataset_dir, pred_json_file)
    
    def process_comprehension(self) -> List[TaskResult]:
        """处理视频理解类任务
        
        需要返回一个TaskResult列表，每个TaskResult包含：
        - task_name: 任务名称，例如 "action_recognition", "video_classification" 等
        - metric: 评估指标，例如 "accuracy", "mAP" 等
        - score: 评估分数
        - task_type: 默认为 TaskType.COMPREHENSION，不需要指定
        
        示例格式：
        return [
            TaskResult(
                task_name="action_recognition",
                metric="accuracy",
                score=0.88
            ),
            TaskResult(
                task_name="video_classification",
                metric="accuracy",
                score=0.92
            )
        ]
        """
        return []
    
    def process_generation(self) -> List[TaskResult]:
        """处理视频生成类任务
        
        需要返回一个TaskResult列表，每个TaskResult包含：
        - task_name: 任务名称，例如 "video_generation", "video_prediction" 等
        - metric: 评估指标，例如 "FVD", "PSNR" 等
        - score: 评估分数
        - task_type: 需要指定为 TaskType.GENERATION
        
        示例格式：
        return [
            TaskResult(
                task_name="video_generation",
                metric="FVD",
                score=45.2,
                task_type=TaskType.GENERATION
            ),
            TaskResult(
                task_name="video_prediction",
                metric="PSNR",
                score=25.8,
                task_type=TaskType.GENERATION
            )
        ]
        """
        return []

# 使用示例
if __name__ == "__main__":
    processor = VideoProcessor(ModalityType.VIDEO, "")
    
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