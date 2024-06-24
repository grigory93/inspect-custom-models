from inspect_ai import Task, eval, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, system_message


def run_eval() -> None:

    @task
    def security_guide(system:str="devops.txt") -> Task:
        return Task(
            dataset=json_dataset("security_guide.jsonl"),
            plan=[system_message(system), generate()],
            scorer=model_graded_fact(
                template="expert.txt", model="openai/gpt-4",
            ),
        )

    tt = security_guide()

    eval([tt], model = "openai/gpt-3.5-turbo")
