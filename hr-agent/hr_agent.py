
import asyncio
import os

from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


# Load environment variables from .env
load_dotenv()

async def main():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Check your .env file."
        )

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key,
    )

    try:
        hr_assistant = AssistantAgent(
            name="hr_assistant",
            system_message="You are an HR assistant.",
            model_client=model_client,
        )

        task = (
            "Write a job description for a Junior Python Developer "
            "in exactly 5 bullet points. Each bullet must be one sentence. "
            "Cover responsibilities, required skills, collaboration, "
            "and learning opportunities."
        )

        await Console(hr_assistant.run_stream(task=task))

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())