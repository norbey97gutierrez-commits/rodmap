from langchain_openai import AzureChatOpenAI

from ..core.config import settings


def get_llm():
    llm = AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_deployment_name,
    )
    return llm


async def get_chat_response(text: str) -> str:
    llm = get_llm()
    response = llm.invoke(text)
    return response.content
