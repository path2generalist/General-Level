from typing import List, Dict, Union, Literal
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    COMPREHENSION = "comprehension"
    GENERATION = "generation"

class ModalityType(Enum):
    IMAGE = "Image"
    VIDEO = "Video"
    AUDIO = "Audio"
    NLP = "NLP"
    THREE_D = "3D"

@dataclass
class TaskResult:
    task_name: str
    metric: str
    score: float
    task_type: TaskType = TaskType.COMPREHENSION  # Default to comprehension task

# Store results for all modalities
ModalityResults = Dict[ModalityType, Dict[TaskType, List[TaskResult]]] 