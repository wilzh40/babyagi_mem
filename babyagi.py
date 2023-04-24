#!/usr/bin/env python3
import os
import time
import logging
import sys
from collections import deque
from typing import Dict, List
import importlib
import openai
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from task_storage import QueueTaskStorage, QueueDAGTaskStorage, NxTaskStorage, Task
from graph import AdjacencyListDAG, NxDAG 
import regex as re
from task import Task
import streamlit as st


# Load logger
import logging
import colorlog

# create a logger
logger = logging.getLogger(__name__)

# set the logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
logger.setLevel(logging.DEBUG)

# Create sidebar to display logs
log_output = st.sidebar.container()

formatter = colorlog.ColoredFormatter(
	"%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
	datefmt=None,
	reset=True,
	log_colors={
		'DEBUG':    'cyan',
		'INFO':     'green',
		'WARNING':  'yellow',
		'ERROR':    'red',
		'CRITICAL': 'red,bg_white',
	},
	secondary_log_colors={},
	style='%'
)

class StreamlitHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        msg = self.format(record)
        log_output.code(msg)

# create a stream handler with the colored formatter
# handler = StreamlitHandler()
handler = logging.StreamHandler()
handler.setFormatter(formatter)
# add the handler to the logger
logger.addHandler(handler)


# Load default environment variables (.env)
load_dotenv()

# Engine configuration

# Model: GPT, LLAMA, HUMAN, etc.
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")).lower()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not (LLM_MODEL.startswith("llama") or LLM_MODEL.startswith("human")):
    assert OPENAI_API_KEY, "\033[91m\033[1m" + "OPENAI_API_KEY environment variable is missing from .env" + "\033[0m\033[0m"

# Table config
RESULTS_STORE_NAME = os.getenv("RESULTS_STORE_NAME", os.getenv("TABLE_NAME", ""))
assert RESULTS_STORE_NAME, "\033[91m\033[1m" + "RESULTS_STORE_NAME environment variable is missing from .env" + "\033[0m\033[0m"

# Run configuration
INSTANCE_NAME = os.getenv("INSTANCE_NAME", os.getenv("BABY_NAME", "BabyAGI"))
COOPERATIVE_MODE = "none"
JOIN_EXISTING_OBJECTIVE = False

# Goal configuation
OBJECTIVE = os.getenv("OBJECTIVE", "")
INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", ""))

# Model configuration
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))

# Extensions support begin

def can_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

DOTENV_EXTENSIONS = os.getenv("DOTENV_EXTENSIONS", "").split(" ")

# Command line arguments extension
# Can override any of the above environment variables
ENABLE_COMMAND_LINE_ARGS = (
    os.getenv("ENABLE_COMMAND_LINE_ARGS", "false").lower() == "true"
)
if ENABLE_COMMAND_LINE_ARGS:
    if can_import("extensions.argparseext"):
        from extensions.argparseext import parse_arguments
        OBJECTIVE, INITIAL_TASK, LLM_MODEL, DOTENV_EXTENSIONS, INSTANCE_NAME, COOPERATIVE_MODE, JOIN_EXISTING_OBJECTIVE = parse_arguments()

# Human mode extension
# Gives human input to babyagi
if LLM_MODEL.startswith("human"):
    if can_import("extensions.human_mode"):
        from extensions.human_mode import user_input_await

# Load additional environment variables for enabled extensions
# TODO: This might override the following command line arguments as well:
#    OBJECTIVE, INITIAL_TASK, LLM_MODEL, INSTANCE_NAME, COOPERATIVE_MODE, JOIN_EXISTING_OBJECTIVE
if DOTENV_EXTENSIONS:
    if can_import("extensions.dotenvext"):
        from extensions.dotenvext import load_dotenv_extensions
        load_dotenv_extensions(DOTENV_EXTENSIONS)


# TODO: There's still work to be done here to enable people to get
# defaults from dotenv extensions, but also provide command line
# arguments to override them

# Extensions support end

print("\033[95m\033[1m"+"\n*****CONFIGURATION*****\n"+"\033[0m\033[0m")
print(f"Name  : {INSTANCE_NAME}")
print(f"Mode  : {'alone' if COOPERATIVE_MODE in ['n', 'none'] else 'local' if COOPERATIVE_MODE in ['l', 'local'] else 'distributed' if COOPERATIVE_MODE in ['d', 'distributed'] else 'undefined'}")
print(f"LLM   : {LLM_MODEL}")

# Check if we know what we are doing
assert OBJECTIVE, "\033[91m\033[1m" + "OBJECTIVE environment variable is missing from .env" + "\033[0m\033[0m"
assert INITIAL_TASK, "\033[91m\033[1m" + "INITIAL_TASK environment variable is missing from .env" + "\033[0m\033[0m"

LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/llama-13B/ggml-model.bin")
if LLM_MODEL.startswith("llama"):
    if can_import("llama_cpp"):
        from llama_cpp import Llama

        print(f"LLAMA : {LLAMA_MODEL_PATH}" + "\n")
        assert os.path.exists(LLAMA_MODEL_PATH), "\033[91m\033[1m" + f"Model can't be found." + "\033[0m\033[0m"

        CTX_MAX = 2048
        THREADS_NUM = 16
        llm = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=CTX_MAX, n_threads=THREADS_NUM,
            use_mlock=True,
        )
        llm_embed = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=CTX_MAX, n_threads=THREADS_NUM,
            embedding=True, use_mlock=True,
        )

        print(
            "\033[91m\033[1m"
            + "\n*****USING LLAMA.CPP. POTENTIALLY SLOW.*****"
            + "\033[0m\033[0m"
        )
    else:
        print(
            "\033[91m\033[1m"
            + "\nLlama LLM requires package llama-cpp. Falling back to GPT-3.5-turbo."
            + "\033[0m\033[0m"
        )
        LLM_MODEL = "gpt-3.5-turbo"

if LLM_MODEL.startswith("gpt-4"):
    print(
        "\033[91m\033[1m"
        + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )

if LLM_MODEL.startswith("human"):
    print(
        "\033[91m\033[1m"
        + "\n*****USING HUMAN INPUT*****"
        + "\033[0m\033[0m"
    )

print("\033[94m\033[1m" + "\n*****OBJECTIVE*****\n" + "\033[0m\033[0m")
print(f"{OBJECTIVE}")

if not JOIN_EXISTING_OBJECTIVE: print("\033[93m\033[1m" + "\nInitial task:" + "\033[0m\033[0m" + f" {INITIAL_TASK}")
else: print("\033[93m\033[1m" + f"\nJoining to help the objective" + "\033[0m\033[0m")

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

# Results storage using local ChromaDB
class DefaultResultsStorage:
    def __init__(self):
        logging.getLogger('chromadb').setLevel(logging.ERROR)
        # Create Chroma collection
        chroma_persist_dir = "chroma"
        chroma_client = chromadb.Client(
            settings=chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=chroma_persist_dir,
            )
        )

        metric = "cosine"
        embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
        self.collection = chroma_client.get_or_create_collection(
            name=RESULTS_STORE_NAME,
            metadata={"hnsw:space": metric},
            embedding_function=embedding_function,
        )

    def add(self, task: Dict, result: Dict, result_id: int, vector: List):

        # Break the function if LLM_MODEL starts with "human" (case-insensitive)
        if LLM_MODEL.startswith("human"):
            return
        # Continue with the rest of the function

        embeddings = [llm_embed.embed(item) for item in vector] if LLM_MODEL.startswith("llama") else None
        if (
            len(self.collection.get(ids=[result_id], include=[])["ids"]) > 0
        ):  # Check if the result already exists
            self.collection.update(
                ids=result_id,
                embeddings=embeddings,
                documents=vector,
                metadatas={"task": task.task_name, "result": result},
            )
        else:
            self.collection.add(
                ids=result_id,
                embeddings=embeddings,
                documents=vector,
                metadatas={"task": task.task_name, "result": result},
            )

    def query(self, query: str, top_results_num: int) -> List[dict]:
        count: int = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_texts=query,
            n_results=min(top_results_num, count),
            include=["metadatas"]
        )
        return [item["task"] for item in results["metadatas"][0]]

# Initialize results storage
results_storage = DefaultResultsStorage()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
if PINECONE_API_KEY:
    if can_import("extensions.pinecone_storage"):
        PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
        assert (
            PINECONE_ENVIRONMENT
        ), "\033[91m\033[1m" + "PINECONE_ENVIRONMENT environment variable is missing from .env" + "\033[0m\033[0m"
        from extensions.pinecone_storage import PineconeResultsStorage
        results_storage = PineconeResultsStorage(OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, LLM_MODEL, LLAMA_MODEL_PATH, RESULTS_STORE_NAME, OBJECTIVE)
        print("\nReplacing results storage: " + "\033[93m\033[1m" +  "Pinecone" + "\033[0m\033[0m")


# Initialize tasks storage
# tasks_storage = QueueTaskStorage()
tasks_storage = QueueDAGTaskStorage(NxDAG)

def openai_call(
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = OPENAI_TEMPERATURE,
    max_tokens: int = 100,
):
    while True:
        try:
            if model.lower().startswith("llama"):
                result = llm(prompt[:CTX_MAX], stop=["### Human"], echo=True, temperature=0.2)
                return result['choices'][0]['text'].strip()
            elif model.lower().startswith("human"):
                return user_input_await(prompt)
            elif not model.lower().startswith("gpt-"):
                # Use completion API
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].text.strip()
            else:
                # Use chat completion API
                messages = [{"role": "system", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                )
                return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "   *** The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.Timeout:
            print(
                "   *** OpenAI API timeout occured. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIError:
            print(
                "   *** OpenAI API error occured. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIConnectionError:
            print(
                "   *** OpenAI API connection error occured. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.InvalidRequestError:
            print(
                "   *** OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.ServiceUnavailableError:
            print(
                "   *** OpenAI API service unavailable. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break


def task_creation_agent(
    objective: str, result: Dict, task_description: str, task_list: List[str]
):
    prompt = f"""
    You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective},
    The last completed task has the result: {result}.
    This result was based on this task description: {task_description}. 
    These are incomplete tasks: {', '.join(task_list)}.
    Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks.
    Return the tasks as an array."""
    response = openai_call(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    return [{"task_name": task_name} for task_name in new_tasks]




import re
def find_function_contents(function_name):
    with open(__file__) as f:
        source_code = f.read()
        pattern = r"def {}\(.*?\):(.+?)^\s*$".format(function_name)
        match = re.search(pattern, source_code, re.DOTALL | re.MULTILINE)
        if match:
            contents = match.group(1).strip()
            return contents
    
def dag_modification_agent(
    objective: str, result: Dict, task_description: str, task_list: List[Task], task_limit=10):
    task_list_str = '\n'.join([str(task) for task in task_list])
    prompt = f"""
    You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective},
    The last completed task has the result: {result}.
    This result was based on this task description: {task_description}. These are incomplete tasks: {task_list_str}.
    Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks in the same format as our tasks, 
    as well as a list of dependencies between tasks in the following format for the purpose of ingestion into networkx Directed Acyclic Graph, where the id is the index of the above returned tasks
    We want to have around at most {task_limit} tasks in the graph.
    Make sure we can parse the tasks by using the following class definition:
    ```
    {Task.get_class_def()}
    ```
    """
    response = openai_call(prompt)
    logger.debug(prompt)
    logger.debug("Response:") 
    logger.debug(response)
    logger.debug( "Parsed response for modifications:")
    logger.debug(Task.from_model_resp(response))
    return Task.from_model_resp(response)


def dag_creation_agent(
    objective: str,  initial_task: str
) -> List[Task]:
    prompt = f"""
    You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective},
    The first task is: {initial_task}.
    Return 2-8 tasks as an array with the following schema: 
    As well as a list of dependencies between tasks in the following format for the purpose of ingestion into networkx Directed Acyclic Graph, where the id is the index of the above returned tasks
    ```
    (task_name: str, difficulty: float from 0.0 to 1.0, dependencies: List[int])
    ```
    Class definition for parsing/unparsing: 
    ```
    {Task.get_class_def()}
    ```
    Example: 
    ("research the history of example subject", difficulty: 0.2, dependencies: [])
    ("implement the code", difficulty: 0.8, dependencies: [0])
    """
    response = openai_call(prompt)
    logger.debug(prompt)
    logger.debug("Response:") 
    logger.debug(response)
    logger.debug( "Parsed response for modifications:")
    logger.debug(Task.from_model_resp(response))
    return Task.from_model_resp(response)


def task_creation_agent_dag(
    objective: str, result: Dict, task_description: str, task_list: List[str], task_dependencies: Dict[str, List[str]], tasks):
    prompt = f"""
    You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective},
    The last completed task has the result: {result}.
    This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(task_list)}.

    These are the dependencies between tasks: {task_dependencies}.

    These are the full metadata for our tasks: {tasks}.

    Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks in the same format as our tasks.
    """
    response = openai_call(prompt)
    print(prompt)
    print(response)
    # new_tasks = response.split("\n") if "\n" in response else [response]
    # return [{"task_name": task_name} for task_name in new_tasks]


def prioritization_agent():
    task_names = tasks_storage.get_task_names()
    next_task_id = tasks_storage.next_task_id()
    prompt = f"""
    You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}.
    Consider the ultimate objective of your team:{OBJECTIVE}.
    Do not remove any tasks. Return the result as a numbered list, like:
    #. First task
    #. Second task
    Start the task list with number {next_task_id}."""
    response = openai_call(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    new_tasks_list = []
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            new_tasks_list.append({"task_id": task_id, "task_name": task_name})
    tasks_storage.replace(new_tasks_list)


# Execute a task based on the objective and five previous tasks
def execution_agent(objective: str, task: str) -> str:
    """
    Executes a task based on the given objective and previous context.

    Args:
        objective (str): The objective or goal for the AI to perform the task.
        task (str): The task to be executed by the AI.

    Returns:
        str: The response generated by the AI for the given task.

    """
    
    context = context_agent(query=objective, top_results_num=5)
    # print("\n*******RELEVANT CONTEXT******\n")
    # print(context)
    prompt = f"""
    You are an AI who performs one task based on the following objective: {objective}\n.
    Take into account these previously completed tasks: {context}\n.
    Your task: {task}\nResponse:"""
    return openai_call(prompt, max_tokens=2000)


# Get the top n completed tasks for the objective
def context_agent(query: str, top_results_num: int):
    """
    Retrieves context for a given query from an index of tasks.

    Args:
        query (str): The query or objective for retrieving context.
        top_results_num (int): The number of top results to retrieve.

    Returns:
        list: A list of tasks as context for the given query, sorted by relevance.

    """
    results = results_storage.query(query=query, top_results_num=top_results_num)
    # print("***** RESULTS *****")
    # print(results)
    return results

# Add the initial task if starting new objective
if not JOIN_EXISTING_OBJECTIVE:
    initial_task = {
        "task_id": tasks_storage.next_task_id(),
        "task_name": INITIAL_TASK
    }
    tasks_storage.append(initial_task)


# Add custom CSS to adjust padding and margins
# st.markdown("""
#     <style>
#         /* Adjust padding and margins of Streamlit components */
#         .stButton, .stTextInput > div > div, .stSelectbox > div > div:first-child, .css-2trqyj:focus, .stTextArea > div > div {
#             padding: 2px 4px;
#             margin: 0px 4px;
#         }
#         .stTextInput > div {
#             margin-top: 0px;
#             margin-bottom: 0px;
#         }
#         .stNumberInput > div > div, .stSelectbox > div > div:first-child {
#             margin-right: 4px;
#         }
#     </style>
# """, unsafe_allow_html=True)
# col1, col2 = st.columns([2, 1])
tab1, tab2, tab3, tab4 = st.tabs(["Tasks/Results", "Graph", "List", "Log"])
graph_placeholder = tab2.empty()
current_task_placeholder = tab1.empty()
current_result_placeholder = tab1.empty()
current_list_placeholder = tab3.empty()

# main iteration loop.
def lets_go(objective: str, initial_task: str):
    initial_tasks = dag_creation_agent(object, initial_task)
    nx_task_storage = NxTaskStorage(OBJECTIVE)
    nx_task_storage.from_tasks(initial_tasks, objective=OBJECTIVE)
    iter = 0
    st.session_state.loop_running = True
    while st.session_state.loop_running:
        # As long as there are tasks in the storage...
        if not nx_task_storage.is_empty():
            iter += 1
            # Print the task list
            print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
            for t in nx_task_storage.get_task_names():
                print(" • "+t)
            # Save the visualization
            nx_task_storage.save_viz("./viz/{}.png".format(iter))
            # Visualize in streamlit.
            # nx_task_storage.st_viz()

            # Step 1: Pull the first incomplete task
            task = nx_task_storage.popleft()
            print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
            print(task.task_name)

            # Send to execution function to complete the task based on the context
            # TODO: Have a more advanced execution agent + save results.
            # TODO: Handle multiple results on the depdenceny tree.
            result = execution_agent(objective, task.task_name)
            print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
            print(result)

            # Step 2: Enrich result and store in the results storage
            # This is where you should enrich the result if needed
            enriched_result = {
                "data": result
            }  
            # extract the actual result from the dictionary
            # since we don't do enrichment currently
            vector = enriched_result["data"]  

            result_id = f"result_{task.id}"

            results_storage.add(task, result, result_id, vector)

            # Visualize this round of tasks before modification.
            with current_task_placeholder.container():
                st.subheader("Current Task: {}".format(task.task_name))
            with current_result_placeholder.container():
                st.subheader("Task Result")
                st.markdown(result, unsafe_allow_html=True)
            with current_list_placeholder.container():
                st.subheader("List of Tasks")
                st.markdown("\n\n".join(nx_task_storage.get_task_names()), unsafe_allow_html=True)
            with graph_placeholder.container():
                st.subheader("DAG")
                nx_task_storage.st_viz()

            # Step 3: Create new tasks and reprioritize task list
            # only the main instance in cooperative mode does that
            # new_tasks = task_creation_agent(
            #     OBJECTIVE,
            #     enriched_result,
            #     task["task_name"],
            #     tasks_storage.get_task_names(),
            # )
            new_tasks = dag_modification_agent(
                objective,
                enriched_result,
                task.task_name,
                nx_task_storage.get_tasks()
            )
            

            # Update our dag
            if new_tasks:
                nx_task_storage.add_tasks(new_tasks)
            else:
                print("No new tasks created")


            # for new_task in new_tasks:
            #     new_task.update({"task_id": tasks_storage.next_task_id()})
            #     tasks_storage.append(new_task)

            # if not JOIN_EXISTING_OBJECTIVE: prioritization_agent()

        # Sleep a bit before checking the task list again
        time.sleep(3) 


# nx_task_storage = NxTaskStorage(OBJECTIVE)
# nx_task_storage.append(Task(INITIAL_TASK, []))
# Streamlit settings.
if 'loop_running' not in st.session_state:
    st.session_state.loop_running = False
# st.markdown("""
#     <style>
#             .block-container {
#                 padding-top: 1rem;
#                 padding-bottom: 0rem;
#                 padding-left: 5rem;
#                 padding-right: 5rem;
#             }
#     </style>
#     """, unsafe_allow_html=True)


def main():
    with st.sidebar:
        st.title("DAGGPT")
        objective = st.text_area(
            "Objective",
            value=st.session_state.get("objective-input", OBJECTIVE),
            key="objective-input",
            height=30
        )
        initial_task = st.text_area(
            "Initial task",
            value=st.session_state.get("init-input", INITIAL_TASK),
            key="init-input",
            height=30
        )

        submit = st.button("Start")
        valid_submission = submit and objective != "" and initial_task != ""
        st.session_state
    if (valid_submission) and not st.session_state.loop_running:
        with st.spinner(text="Looping..."):
            st.cache_data.clear()
            lets_go(objective=objective, initial_task=initial_task)
            st.experimental_rerun()


if __name__ == "__main__":
    main()
