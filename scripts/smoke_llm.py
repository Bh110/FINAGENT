"""Integration check (needs Ollama): python -m scripts.smoke_llm"""
from llm.model import llm, llm_status

print(llm_status())
print(llm.invoke("Explain in one sentence what an investment risk is.").content)
