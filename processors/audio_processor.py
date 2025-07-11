from typing import List
from utils.data_types import ModalityType, TaskType, TaskResult
from utils.base_processor import BaseModalityProcessor

class AudioProcessor(BaseModalityProcessor):
    """音频模态处理器"""
    def __init__(self, modality: ModalityType, dataset_dir: str, pred_json_file: str):
        super().__init__(modality, dataset_dir, pred_json_file)
    
    def process_comprehension(self) -> List[TaskResult]:
        """处理音频理解类任务
        
        需要返回一个TaskResult列表，每个TaskResult包含：
        - task_name: 任务名称，例如 "speech_recognition", "audio_classification" 等
        - metric: 评估指标，例如 "WER", "accuracy" 等
        - score: 评估分数
        - task_type: 默认为 TaskType.COMPREHENSION，不需要指定
        
        示例格式：
        return [
            TaskResult(
                task_name="speech_recognition",
                metric="WER",
                score=0.15
            ),
            TaskResult(
                task_name="audio_classification",
                metric="accuracy",
                score=0.92
            )
        ]
        """
        return []
    
    def process_generation(self) -> List[TaskResult]:
        """处理音频生成类任务
        
        需要返回一个TaskResult列表，每个TaskResult包含：
        - task_name: 任务名称，例如 "speech_synthesis", "audio_generation" 等
        - metric: 评估指标，例如 "MOS", "FAD" 等
        - score: 评估分数
        - task_type: 需要指定为 TaskType.GENERATION
        
        示例格式：
        return [
            TaskResult(
                task_name="speech_synthesis",
                metric="MOS",
                score=4.2,
                task_type=TaskType.GENERATION
            ),
            TaskResult(
                task_name="audio_generation",
                metric="FAD",
                score=12.5,
                task_type=TaskType.GENERATION
            )
        ]
        """
        return []

# 使用示例
if __name__ == "__main__":
    processor = AudioProcessor(ModalityType.AUDIO, "")
    
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