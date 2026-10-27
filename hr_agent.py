
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main():
    # Read the API key from the environment variable.
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Set it before running the script."
        )

    # Create the OpenAI model client.
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key,
    )

    try:
        # Create the HR assistant.
        hr_assistant = AssistantAgent(
            name="hr_assistant",
            system_message="You are an HR assistant.",
            model_client=model_client,
        )

        # Request exactly five bullet points.
        task = (
            "Write a job description for a Junior Python Developer "
            "in exactly 5 bullet points. Each bullet must be one sentence. "
            "Cover responsibilities, required skills, collaboration, "
            "and learning opportunities."
        )

        # Run the agent and display its response.
        await Console(hr_assistant.run_stream(task=task))

    finally:
        # Release the model client's resources.
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
