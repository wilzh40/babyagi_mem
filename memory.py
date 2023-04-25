from thinkgpt.llm import ThinkGPT
from dotenv import load_dotenv
load_dotenv()
llm = ThinkGPT(model_name="gpt-3.5-turbo")
# Make the llm object learn new concepts
# llm.memorize(['DocArray is a library for representing, sending and storing multi-modal data.'])
# print(llm.predict('what is DocArray ?', remember=llm.remember('DocArray definition')))


from thinkgpt.llm import ThinkGPT
import streamlit as st


MAX_MEMORY_ITEM_SIZE = 1000
# Hints inspired by https://github.com/muellerberndt/micro-gpt.
SUMMARY_HINT = "Do your best to retain all semantic information including tasks performed"\
    "by the agent, website content, important data points and hyper-links.\n"
EXTRA_SUMMARY_HINT = "If the text contains information related to the topic: '{summarizer_hint}'"\
    "then include it. If not, write a standard summary."

class SimpleMemory:
    def __init__(self):
        self.llm = ThinkGPT(model_name="gpt-3.5-turbo")
        self.events = []

    def memorize(self, event: str):
        self.events.append(event)
        self.llm.memorize(event)

    def memorize_task(self, task, result):
        self.memorize(f"TASK:{task}\nRESULT:{result}")

    def remember(self, event: str):
        remembered = self.llm.remember(event, limit=10, sort_by_order=True)
        return remembered

    def enrich_result(self, event:str):
        return self.llm.summarize(event, instruction_hint= "SUMMARY_HINT")

    def render_viz(self):
        events_bullets = "\n".join([f"- {e}" for e in self.events])
        st.write(events_bullets)

# TODO add long-term memory (ie saving into files and loading those files in and summarizing them).

class ProceduralMemory:
    def __init__(self):
        self.llm = ThinkGPT(model_name="gpt-3.5-turbo")
        self.rules = []

    def memorize(self, rule):
        self.rules.append(rule)
        self.llm.memorize(rule)

class SemanticMemory:
    def __init__(self):
        self.llm = ThinkGPT(model_name="gpt-3.5-turbo")
        self.facts = []

    def memorize(self, fact):
        self.facts.append(fact)
        self.llm.memorize(fact)
    
    def memorize_file(self, arg):
        with open(arg, "r") as f:
            file_content = self.llm.chunked_summarize(
                f.read(), max_memory_item_size,
                instruction_hint=SUMMARY_HINT +
                    EXTRA_SUMMARY_HINT.format(objective=objective))
            self.llm.memorize(f"{mem}{file_content}")

class EpisodicMemory:
    def __init__(self):
        self.events = []

    def remember_event(self, event):
        self.events.append(event)

# Example usage
pm = ProceduralMemory()
pm.memorize("in tunisian, I did not eat is \"ma khditech\"")
pm.memorize("I did not work is \"ma khdemtech\"")

sm = SemanticMemory()
sm.memorize("in tunisian, I studied is \"9rit\"")

em = EpisodicMemory()
em.remember_event("I went to the beach yesterday")

# To retrieve a memorized rule or fact:
print(pm.rules)
print(sm.facts)
print(em.events)



