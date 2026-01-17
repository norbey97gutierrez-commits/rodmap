from src.domain.entities.schemas import IntentionResponse


def build_structured_llm(llm):
    return llm.with_structured_output(IntentionResponse)
