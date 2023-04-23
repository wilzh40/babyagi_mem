
from abc import ABC, abstractmethod
from typing import List, Dict
from collections import deque
# Task storage supporting only a single instance of BabyAGI
class TaskStorage(ABC):
    @abstractmethod
    def append(self, task: Dict):
        pass

    @abstractmethod
    def replace(self, tasks: List[Dict]):
        pass

    @abstractmethod
    def popleft(self):
        pass

    @abstractmethod
    def is_empty(self):
        pass

    @abstractmethod
    def next_task_id(self):
        pass

    @abstractmethod
    def get_task_names(self):
        pass

class QueueTaskStorage(TaskStorage):
    def __init__(self):
        self.tasks = deque([])
        self.task_id_counter = 0

    def append(self, task: Dict):
        self.tasks.append(task)

    def replace(self, tasks: List[Dict]):
        self.tasks = deque(tasks)

    def popleft(self):
        return self.tasks.popleft()

    def is_empty(self):
        return False if self.tasks else True

    def next_task_id(self):
        self.task_id_counter += 1
        return self.task_id_counter

    def get_task_names(self):
        return [t["task_name"] for t in self.tasks]

class DAGTaskStorage(TaskStorage):
    def __init__(self):
        pass
