
from abc import ABC, abstractmethod
from typing import List, Dict, TypeVar
from collections import deque
# Task storage supporting only a single instance of BabyAGI
from graph import DAG, AdjacencyListDAG
from datetime import datetime
T = TypeVar('T', bound=DAG)

# class Task:
#     def __init__(self, task_name, task_params):
#         self.task_name = task_name
#         self.task_params = task_params
#         self.dependencies = set()
#         self.status = "pending"
#         self.result = None
#         self.priority = 0
#         self.creation_timestamp = datetime.now()
#         self.id = self.creation_timestamp

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

    def next_task_id(self):
        self.task_id_counter += 1
        return self.task_id_counter

    @abstractmethod
    def get_task_names(self):
        pass

class QueueTaskStorage(TaskStorage):
    def __init__(self):
        self.task_id_counter = 0
        self.tasks = deque([])

    def append(self, task: Dict):
        self.tasks.append(task)

    def replace(self, tasks: List[Dict]):
        self.tasks = deque(tasks)

    def popleft(self):
        return self.tasks.popleft()

    def is_empty(self):
        return False if self.tasks else True

    def get_task_names(self):
        return [t["task_name"] for t in self.tasks]

# The DAG will store the task IDs. Aiming to have feature parity with the old QueueTaskStorage but using a DAG as the underlying data structure.
# TODO: Somehow use the latent structure of the dag to make the task selection more efficient and prioritzie noes.
class QueueDAGTaskStorage(TaskStorage):
    def __init__(self, dag_class: type[DAG]):
        super().__init__()
        self.dag = dag_class()
        self.task_id_counter = 0
        self.tasks_dict = {}

    def append(self, task: Dict):
        task_id = self.next_task_id()
        task["id"] = task_id
        task["creation_timestamp"] = datetime.now()
        self.tasks_dict[task_id] = task
        # Choose a random leaf and add the task as a successor.
        leaves = list(self.dag.get_leaves())
        if len(leaves) > 0:
            self.dag.add_edge(leaves[0], task_id)
        else:
            self.dag.add_node(task_id)
    
    def replace(self, tasks: DAG):
        self.tasks = tasks

    def popleft(self):
        order = self.dag.topological_sort()
        node = order[0]
        self.dag.remove_node(node)
        return self.tasks_dict[node]

    def is_empty(self):
        return False if self.dag.nodes else True

    def get_task_names(self):
        return [self.tasks_dict[i]["task_name"] for i in self.dag.nodes]