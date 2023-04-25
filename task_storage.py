
from abc import ABC, abstractmethod
from typing import List, Dict, TypeVar, Tuple, Hashable
from collections import deque
# Task storage supporting only a single instance of BabyAGI
from graph import DAG, AdjacencyListDAG, NxDAG
from datetime import datetime
import networkx as nx
import inspect
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from task import Task

T = TypeVar('T', bound=DAG)

# class TaskStorage(ABC):
#     @abstractmethod
#     def append(self, task: Dict):
#         pass

#     @abstractmethod
#     def replace(self, tasks: List[Dict]):
#         pass

#     @abstractmethod
#     def popleft(self):
#         pass

#     @abstractmethod
#     def is_empty(self):
#         pass

#     def next_task_id(self):
#         self.task_id_counter += 1
#         return self.task_id_counter

#     @abstractmethod
#     def get_task_names(self):
#         pass

# class QueueTaskStorage(TaskStorage):
#     def __init__(self):
#         self.task_id_counter = 0
#         self.tasks = deque([])

#     def append(self, task: Dict):
#         self.tasks.append(task)

#     def replace(self, tasks: List[Dict]):
#         self.tasks = deque(tasks)

#     def popleft(self):
#         return self.tasks.popleft()

#     def is_empty(self):
#         return False if self.tasks else True

#     def get_task_names(self):
#         return [t["task_name"] for t in self.tasks]

# # The DAG will store the task IDs. Aiming to have feature parity with the old QueueTaskStorage but using a DAG as the underlying data structure.
# # TODO: Somehow use the latent structure of the dag to make the task selection more efficient and prioritzie noes.
# class QueueDAGTaskStorage(TaskStorage):
#     def __init__(self, dag_class: type[DAG]):
#         super().__init__()
#         self.dag = dag_class()
#         self.task_id_counter = 0
#         self.tasks_dict = {}

#     def append(self, task: Dict):
#         task_id = self.next_task_id()
#         task["id"] = task_id
#         task["creation_timestamp"] = datetime.now()
#         self.tasks_dict[task_id] = task
#         # Choose a random leaf and add the task as a successor.
#         leaves = list(self.dag.get_leaves())
#         if len(leaves) > 0:
#             self.dag.add_edge(leaves[0], task_id)
#         else:
#             self.dag.add_node(task_id)
    
#     def replace(self, tasks: DAG):
#         self.tasks = tasks

#     def popleft(self):
#         order = self.dag.topological_sort()
#         node = order[0]
#         self.dag.remove_node(node)
#         return self.tasks_dict[node]

#     def is_empty(self):
#         return False if self.dag.get_nodes() else True

#     def get_task_names(self):
#         return [self.tasks_dict[i]["task_name"] for i in self.dag.get_nodes()]

#     def get_task_dependencies(self):
#         return {i : self.dag.get_predecessors(i) for i in self.dag.get_nodes()}

#     def save_viz(self):
#         mapping = {k: v["task_name"] for k, v in self.tasks_dict.items()}
#         self.dag.print(mapping=mapping)

# import re
# def find_class_contents(class_name):
#     print(__file__)
#     with open(__file__) as f:
#         source_code = f.read()
#         pattern = r"class\s+{}\s*\((.*?)\)\s*(?::\s*(.+?))?(?=class|\Z)".format(class_name)
#         match = re.search(pattern, source_code, re.DOTALL | re.MULTILINE)
#         if match:
#             return match.group(2).strip()
        


# Breaking the absraction here, still a bit rusty from Python.
class NxTaskStorage():
    graph: nx.DiGraph
    def __init__(self, objective: str):
        self.task_id_counter = -1
        self.tasks_dict = {}
        self.graph = nx.DiGraph()
        self.objective_node = objective
        self.tasks_dict[0] = Task(self.objective_node, objective,  {})
        self.graph.add_node(self.objective_node)

    def get_tasks(self):
        return self.tasks_dict.values()

    def from_tasks(self, tasks: List[Task], objective: str):
        self.graph = nx.DiGraph()
        self.objective_node = -1
        self.graph.add_node(self.objective_node)
        self.tasks_dict[self.objective_node] = Task(self.objective_node, objective, {})

        for task in tasks:
            self.tasks_dict[task.id] = task
            self.graph.add_node(task.id)
            for dep in task.dependencies:
                self.graph.add_edge(dep, task.id)
            self.graph.add_edge(task.id, self.objective_node)

    def add_tasks(self, tasks: List[Task]):
        for task in tasks:
            self.tasks_dict[task.id] = task
            self.graph.add_node(task.id)
            for dep in task.dependencies:
                self.graph.add_edge(task.id, dep)
            self.graph.add_edge(task.id, self.objective_node)
        
    def is_empty(self):
        # Other than the objective.
        return self.graph.number_of_nodes() <= 1

    def popleft(self) -> Task:
        order = list(nx.topological_sort(self.graph))
        node = order[0]
        self.graph.remove_node(node)
        return self.tasks_dict[node]

    def next_task_id(self):
        self.task_id_counter += 1
        return self.task_id_counter 

    # def append(self, task: Task) -> None:
    #     task_id = self.next_task_id()
    #     task.id = task_id
    #     task.creation_timestamp = datetime.now()
    #     self.tasks_dict[task_id] = task
    #     self.graph.add_edge(self.objective_node, task_id)

    def append_multiple(self, tasks: Dict[Hashable, Task], edges: List[Tuple[int, int]]):
        self.tasks_dict.update(tasks)
        self.graph.add_edges_from(edges)

    def get_task_depdendencies(self):
        task_order = list(nx.topological_sort(self.graph))
        return {i : list(self.graph.predecessors(i)) for i in task_order}

    def get_tasks(self) -> List[Task]:
        task_order = list(nx.topological_sort(self.graph))
        return [self.tasks_dict[i] for i in task_order]

    def get_task_names(self) -> List[str]:
        task_order = list(nx.topological_sort(self.graph))
        return [self.tasks_dict[i].task_name for i in task_order]

    def save_viz(self, path="./viz/current_task.png"):
        mapping = {k: v.task_name for k, v in self.tasks_dict.items()}
        relabeled_graph = nx.relabel_nodes(self.graph, mapping)
        agraph = nx.drawing.nx_agraph.to_agraph(relabeled_graph)
        agraph.draw(path, prog='dot')

    # def _add_expand_delete_buttons(self, node) -> None:
    #     st.sidebar.subheader(node)
    #     cols = st.sidebar.columns(2)
    #     cols[1].button(
    #         label="Delete", 
    #         on_click=self._delete_node,
    #         type="primary",
    #         key=f"delete_{node}",
    #         # pass on to _delete_node
    #         args=(node,)
    #     )
    
    def st_viz(self):
        # mapping = {k: v.task_name for k, v in self.tasks_dict.items()}
        # relabeled_graph = nx.relabel_nodes(self.graph, mapping)
        # agraph = nx.drawing.nx_agraph.to_agraph(relabeled_graph)

        selected = st.session_state.get("last_expanded")
        COLOR = "cyan"
        FOCUS_COLOR = "red"
        vis_nodes = [
            Node(
                id=n, 
                label=self.tasks_dict[n].task_name + f"({self.tasks_dict[n].difficulty})",
                # a little bit bigger if selected, scaled by difficulty.
                size=self.tasks_dict[n].difficulty*20 + 10 + 10 * (n==selected),
                borderWidth=1,
                # a different color if selected
                color=COLOR if n != selected else FOCUS_COLOR
            ) 
            for n in self.graph.nodes]
        vis_edges = [Edge(source=a, target=b) for a, b in self.graph.edges]
        config = Config(width="100%",
                        height=600,
                        directed=True, 
                        physics=True,
                        hierarchical=False,
                        )
        # returns a node if clicked, otherwise None
        clicked_node = agraph(nodes=vis_nodes, 
                        edges=vis_edges, 
                        config=config)


        # Define the node size based on the task difficulty
        # sizes = [task.difficulty*200 + 50 for task in self.tasks_dict.values()]

        # Render the agraph visualization
        # agraph(nodes=self.graph.nodes, edges=self.graph.edges, config=config)
        # if clicked, update the sidebar with a button to create it
        # if clicked_node is not None:
        #     self._add_expand_delete_buttons(clicked_node)

    
if __name__ == '__main__':
    print("hello")
    print(Task.get_class_def())
    task_storage = QueueDAGTaskStorage(NxDAG)
    task_storage.append({"task_name": "task1", "task_params": {}})
    task_storage.append({"task_name": "task2", "task_params": {}})
    # task_storage.save_viz()

    task_storage = NxTaskStorage()
    task_storage.append_multiple({0: Task("Create a task list", {}), 
                                  1: Task("Figure life out", {})}, [(0, 1)])
    task_storage.save_viz()
    print(task_storage.get_task_depdendencies())
    print(task_storage.get_tasks())
    print(task_storage.get_task_names())
    # print(task_storage.to_string())