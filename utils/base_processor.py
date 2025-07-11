from typing import List
from .data_types import ModalityType, TaskType, TaskResult

"""Base modality processor"""

class BaseModalityProcessor:
    def __init__(self, modality: ModalityType, 
                 dataset_dir: str, 
                 pred_json_file: str):
        self.modality = modality
        self.dataset_dir = dataset_dir
        self.pred_json_file = pred_json_file
    
    def process_comprehension(self) -> List[TaskResult]:
        """Process comprehension tasks, optional implementation"""
        return []
    
    def process_generation(self) -> List[TaskResult]:
        """Process generation tasks, optional implementation"""
        return []
    
    def process(self) -> List[TaskResult]:
        """Process tasks without type distinction (e.g., NLP tasks)"""
        return [] 