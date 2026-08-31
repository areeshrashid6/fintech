"""
chains.py
Builds the ChatOpenAI model, the reusable LLMChain for structured JSON
analysis, a raw SystemMessage/HumanMessage/AIMessage demo, and the
streaming generator used for the live narrative recommendation.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from src.prompts import JSON_CHAT_TEMPLATE, NARRATIVE_CHAT_TEMPLATE, SYSTEM_PROMPT


def get_llm(api_key: str, model: str, temperature: float = 0.4, streaming: bool = False) -> ChatOpenAI:
    """Create a ChatOpenAI instance connected to the given model."""
    return ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=temperature,
        streaming=streaming,
    )


def build_analysis_chain(llm: ChatOpenAI):
    """
    Reusable LLMChain-style pipeline (LCEL) that turns the JSON_CHAT_TEMPLATE
    + user data into a raw JSON string. Equivalent in spirit to
    `LLMChain(llm=llm, prompt=JSON_CHAT_TEMPLATE)`.
    """
    return JSON_CHAT_TEMPLATE | llm | StrOutputParser()


def run_analysis(llm: ChatOpenAI, inputs: dict) -> str:
    """Run the structured-JSON analysis chain and return the raw text output."""
    chain = build_analysis_chain(llm)
    return chain.invoke(inputs)


def stream_recommendations(llm: ChatOpenAI, inputs: dict):
    """
    Stream the narrative recommendation chunk by chunk, for use with
    st.write_stream() to produce a live typing effect.
    """
    messages = NARRATIVE_CHAT_TEMPLATE.format_messages(**inputs)
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def message_demo(inputs: dict) -> list:
    """
    Demonstrates how SystemMessage, HumanMessage, and AIMessage represent a
    conversation turn. Not used for generation — purely illustrative, and
    shown in the UI to satisfy the assignment's message-handling requirement.
    """
    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    human_msg = HumanMessage(
        content=(
            f"My monthly income is {inputs.get('monthly_income')} and my total "
            f"expenses are {inputs.get('total_expenses')}. My goal is "
            f"'{inputs.get('financial_goal')}'."
        )
    )
    ai_msg = AIMessage(
        content=(
            "Thanks — I'll analyse your income, expenses, and goal, and return "
            "a structured, educational budget summary."
        )
    )
    return [system_msg, human_msg, ai_msg]
