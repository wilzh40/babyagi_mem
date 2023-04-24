import inspect
import re
import unittest
class Task:
    def __init__(self, id, task_name, task_params, dependencies=[], status="pending", result=None, priority=0, difficulty=0, creation_timestamp=None):
        self.id = id
        self.task_name = task_name
        self.task_params = task_params
        self.status = "pending"
        self.result = None
        self.priority = 0
        self.difficulty = 0
        self.dependencies = dependencies
        self.creation_timestamp = datetime.now()
        
 
    def __str__(self) -> str:
            dep_str = "[" + ', '.join(str(dep) for dep in self.dependencies) + "]" if len(self.dependencies) > 0 else "[]"
            return f"{self.id}. {self.task_name} (difficulty: {self.difficulty}, dependencies: {dep_str})"

    def __repr__(self):
        return f"Task(id={self.id}, task_name={self.task_name}, task_params={self.task_params}, status={self.status}, result={self.result}, priority={self.priority}, difficulty={self.difficulty}, dependencies={self.dependencies}, creation_timestamp={self.creation_timestamp})"
        
    @classmethod
    def from_model_resp(cls, string):
        tasks = []
        pattern = r'(?P<id>\d+)\.\s+(?P<task_name>.*?)\s+\(difficulty:\s+(?P<difficulty>[\d\.]+),\s+dependencies:\s+(?P<dependencies>\[.*?\]|none)\)'
        for match in re.finditer(pattern, string):
            task_id = int(match.group('id'))
            task_name = match.group('task_name')
            difficulty = float(match.group('difficulty'))
            dependencies = match.group('dependencies')
            if dependencies != "none":
                dependencies = [int(dep_id) for dep_id in re.findall(r'\d+', dependencies)]
            else:
                dependencies = []
            task = cls(task_id, task_name, {}, difficulty=difficulty, dependencies=dependencies)
            tasks.append(task)
        return tasks
        
    @classmethod
    def get_class_def(cls):
        return inspect.getsource(Task)



import unittest
from datetime import datetime
from task import Task

class TestTask(unittest.TestCase):

    def test_from_model_resp_and_str_isomorphism(self):
        task_str = "1. Task1 (difficulty: 0.5, dependencies: [2])\n2. Task2 (difficulty: 0.3, dependencies: none)\n3. Task3 (difficulty: 0.7, dependencies: [1, 2])"
        tasks = Task.from_model_resp(task_str)
        for i, task in enumerate(tasks):
            task_str = str(task)
            expected_str = f"{i+1}. {task.task_name} (difficulty: {task.difficulty}, dependencies: {task.dependencies})"
            print(expected_str)
            self.assertEqual(task_str, expected_str) 