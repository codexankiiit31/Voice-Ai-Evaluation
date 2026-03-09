"""
LLM inference module using LangChain + Ollama.

Provides functions to load an Ollama LLM and generate responses
to questions via a prompt template chain.
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Global LLM cache: {model_name: OllamaLLM}
_llm_cache: dict = {}

# Prompt template for Q&A
_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["question"],
    template="Answer in 1-2 sentences. Question: {question} Answer:",
)


def load_llm(
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
) -> OllamaLLM:
    """
    Load and cache an Ollama LLM instance.

    Args:
        model: The Ollama model name (e.g. 'llama3', 'mistral').
        base_url: The Ollama server URL.

    Returns:
        A configured OllamaLLM instance.
    """
    if model not in _llm_cache:
        _llm_cache[model] = OllamaLLM(
            model=model,
            base_url=base_url,
            temperature=0,
            num_predict=256,
        )
    return _llm_cache[model]


def generate_response(prompt: str, model: str = "llama3.2") -> str:
    """
    Generate an LLM response for a given question.

    Uses a PromptTemplate + OllamaLLM chain to produce a concise answer.

    Args:
        prompt: The question text to answer.
        model: The Ollama model name to use.

    Returns:
        The LLM-generated answer string, stripped of whitespace.
    """
    llm = load_llm(model)
    chain = _PROMPT_TEMPLATE | llm
    response = chain.invoke({"question": prompt})
    return response.strip()
