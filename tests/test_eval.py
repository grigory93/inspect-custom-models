import os

import pytest
from inspect_ai import Task, eval, task
from inspect_ai.dataset import example_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, system_message

SYSTEM_MESSAGE = """
You are a computer security expert and you are tasked with
providing concise response to the following questions.
Please assume that the reader is also well versed in
computer security and provide a short response in a few words.
"""

def test_env() -> None:
    assert os.getenv("OPENAI_API_KEY"), "No openai API key found"
    assert os.getenv("AI21_API_KEY"), "No AI21 API key found"

def run_eval(model:str, model_eval:str = "openai/gpt-3.5-turbo") -> None:

    @task
    def security_guide(model_eval:str) -> Task:
        return Task(
            dataset=example_dataset("security_guide"),
            plan=[system_message(SYSTEM_MESSAGE), generate()],
            scorer=model_graded_fact(model=model_eval),
        )

    tt = security_guide(model_eval=model_eval)

    eval([tt], model = model)



def test_run_eval_with_openai() -> None:
    try:
        model = "openai/gpt-3.5-turbo"
        run_eval(model)
        assert True
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"run_eval with model {model} raised an exception: {e}")


def test_run_eval_with_ai21() -> None:
    try:
        model = "ai21/jamba-instruct-preview"
        run_eval(model)
        assert True
    except Exception as e: # noqa: BLE001
        pytest.fail(f"run_eval with model {model} raised an excpetion: {e}")
