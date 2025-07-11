from typing import List
from utils.data_types import ModalityType, TaskType, TaskResult
from utils.base_processor import BaseModalityProcessor

class ThreeDProcessor(BaseModalityProcessor):
    """3D模态处理器"""
    def __init__(self, modality: ModalityType, dataset_dir: str, pred_json_file: str):
        super().__init__(modality, dataset_dir, pred_json_file)
    
    def process_comprehension(self) -> List[TaskResult]:
        """处理3D理解类任务
        
        需要返回一个TaskResult列表，每个TaskResult包含：
        - task_name: 任务名称，例如 "3d_object_detection", "point_cloud_segmentation" 等
        - metric: 评估指标，例如 "mAP", "IoU" 等
        - score: 评估分数
        - task_type: 默认为 TaskType.COMPREHENSION，不需要指定
        示例格式：
        return [
            TaskResult(
                task_name="3d_object_detection",
                metric="mAP",
                score=0.76
            ),
            TaskResult(
                task_name="point_cloud_segmentation",
                metric="IoU",
                score=0.82
            )
        ]
        """
        return []
    
    def process_generation(self) -> List[TaskResult]:
        """处理3D生成类任务
        
        需要返回一个TaskResult列表，每个TaskResult包含：
        - task_name: 任务名称，例如 "3d_reconstruction", "mesh_generation" 等
        - metric: 评估指标，例如 "CD", "F1" 等
        - score: 评估分数
        - task_type: 这里需要指定为 TaskType.GENERATION
        
        示例格式：
        return [
            TaskResult(
                task_name="3d_reconstruction",
                metric="CD",
                score=0.15,
                task_type=TaskType.GENERATION
            ),
            TaskResult(
                task_name="mesh_generation",
                metric="F1",
                score=0.88,
                task_type=TaskType.GENERATION
            )
        ]
        """
        return []

# 使用示例
if __name__ == "__main__":
    processor = ThreeDProcessor(ModalityType.THREE_D, "")
    
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