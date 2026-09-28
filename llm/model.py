from langchain_ollama import ChatOllama

import config

_common = dict(
    model=config.LLM_MODEL,
    temperature=config.LLM_TEMPERATURE,
    num_ctx=config.LLM_NUM_CTX,
    client_kwargs={"timeout": config.LLM_TIMEOUT_S},
)

# Free-text model (follow-up chat, test_llm.py)
llm = ChatOllama(**_common)

# JSON-constrained model used by the Analysis Agent
json_llm = ChatOllama(format="json", **_common)


def llm_status():
    """(ok, message) - checks that Ollama is reachable and the model is pulled."""
    try:
        import ollama
        listing = ollama.Client().list()
        names = [getattr(m, "model", None) or getattr(m, "name", "") for m in listing.models]
        if any(n == config.LLM_MODEL or n.split(":")[0] == config.LLM_MODEL for n in names):
            return True, f"{config.LLM_MODEL} ready"
        return False, f"Ollama is running but '{config.LLM_MODEL}' is not pulled " \
                      f"(run: ollama pull {config.LLM_MODEL})"
    except Exception as e:
        return False, f"Ollama unreachable ({type(e).__name__}). Start it with: ollama serve"
