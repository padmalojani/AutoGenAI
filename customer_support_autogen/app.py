import asyncio
import os

import streamlit as st
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
)
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

# -------------------------------
# Streamlit configuration
# -------------------------------
st.set_page_config(
    page_title="AutoGen Customer Support",
    page_icon="🎧",
    layout="centered",
)

st.title("🎧 AI Customer Support")
st.caption("Five specialized AutoGen agents collaborate to solve your issue.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is missing. Add it to your .env file.")
    st.stop()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def create_support_team():
    """Create five specialized customer support agents."""

    model_client = OpenAIChatCompletionClient(
        model=MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    rag_agent = AssistantAgent(
        name="RAGSupport",
        model_client=model_client,
        description="Handles FAQs, company policies, product manuals "
                    "and knowledge-base questions.",
        system_message=(
            "You are the RAG Support Specialist. "
            "Answer using only information supplied in the conversation. "
            "No company documents are connected yet, so do not pretend "
            "to have searched a knowledge base. State what information "
            "would be needed when appropriate. Keep your response concise."
        ),
    )

    web_agent = AssistantAgent(
        name="WebResearch",
        model_client=model_client,
        description="Handles requests requiring current external information.",
        system_message=(
            "You are the Web Research Specialist. "
            "Identify what external research would help. No web search "
            "tool is connected in this starter version, so never claim "
            "you searched the internet or verified live information. "
            "Give cautious guidance based on available context."
        ),
    )

    care_agent = AssistantAgent(
        name="CustomerCare",
        model_client=model_client,
        description="Handles complaints, orders, refunds, billing "
                    "and customer communication.",
        system_message=(
            "You are the Customer Care Specialist. "
            "Help with orders, billing, refunds and complaints. "
            "Ask for missing non-sensitive details when necessary. "
            "Never claim to issue a refund, change an order, or access "
            "a customer account without an actual connected tool."
        ),
    )

    technical_agent = AssistantAgent(
        name="TechnicalSupport",
        model_client=model_client,
        description="Diagnoses software, product and technical problems.",
        system_message=(
            "You are the Technical Support Specialist. "
            "Diagnose technical problems step by step. "
            "Give safe, practical troubleshooting instructions. "
            "Ask for error messages or product details if needed."
        ),
    )

    supervisor = AssistantAgent(
        name="Supervisor",
        model_client=model_client,
        description="Routes requests to specialists and produces "
                    "the final customer-facing answer.",
        system_message=(
            "You are the Customer Support Supervisor. "
            "At the beginning, briefly classify the customer's issue "
            "and identify the specialist or specialists needed. "
            "Let relevant specialists contribute. Do not require all "
            "specialists to participate. After receiving useful input, "
            "combine the findings into one accurate, friendly answer. "
            "If the issue is ambiguous, ask a clarifying question. "
            "Never invent company policies or claim actions were taken. "
            "Your final message MUST start with CUSTOMER_RESPONSE: "
            "and contain the complete answer for the customer."
        ),
    )

    selector_prompt = """
You coordinate a customer support team.

Available agents:
{roles}

Conversation:
{history}

Rules:
1. On the first turn, select Supervisor.
2. After Supervisor classifies the issue, select only a relevant
   specialist: RAGSupport, WebResearch, CustomerCare, or TechnicalSupport.
3. A specialist should contribute only when their expertise is relevant.
4. After the needed specialist(s) have contributed, select Supervisor.
5. Supervisor should deliver one consolidated answer beginning with
   CUSTOMER_RESPONSE:.
6. If Supervisor has already delivered CUSTOMER_RESPONSE:, do not
   select another speaker.
7. Select exactly one agent name from {participants}.
"""

    termination = (
        TextMentionTermination("CUSTOMER_RESPONSE:")
        | MaxMessageTermination(8)
    )

    team = SelectorGroupChat(
        participants=[
            supervisor,
            rag_agent,
            web_agent,
            care_agent,
            technical_agent,
        ],
        model_client=model_client,
        selector_prompt=selector_prompt,
        termination_condition=termination,
        max_turns=8,
    )

    return team, model_client


async def process_query(question: str):
    team, model_client = create_support_team()

    try:
        result = await team.run(task=question)
        messages = result.messages

        # Prefer the supervisor's final answer.
        final_answer = None

        for message in reversed(messages):
            if (
                isinstance(message, TextMessage)
                and message.source == "Supervisor"
                and "CUSTOMER_RESPONSE:" in message.content
            ):
                final_answer = message.content.split(
                    "CUSTOMER_RESPONSE:", 1
                )[1].strip()
                break

        if not final_answer:
            final_answer = (
                "The team could not complete a reliable answer. "
                "Please clarify your issue or try again."
            )

        transcript = []
        for message in messages:
            if isinstance(message, TextMessage):
                transcript.append(
                    (message.source, message.content)
                )

        return final_answer, transcript

    finally:
        await model_client.close()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous customer conversations.
for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

        if item.get("transcript"):
            with st.expander("View agent group-chat discussion"):
                for speaker, content in item["transcript"]:
                    st.markdown(f"**{speaker}:**")
                    st.write(content)

question = st.chat_input("Describe your issue or ask a question...")

if question:
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("The support agents are collaborating..."):
            try:
                answer, transcript = asyncio.run(
                    process_query(question)
                )

                st.markdown(answer)

                with st.expander("View agent group-chat discussion"):
                    for speaker, content in transcript:
                        st.markdown(f"**{speaker}:**")
                        st.write(content)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "transcript": transcript,
                })

            except Exception as exc:
                st.error(
                    "Unable to process your request. "
                    "Check your API key, model access and connection."
                )
                st.caption(str(exc))
