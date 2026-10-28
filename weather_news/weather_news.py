import os
import asyncio
from datetime import date

from dotenv import load_dotenv
from ddgs import DDGS

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


load_dotenv()

# 1. Search tool for today's weather news
def search_weather_news(city: str) -> str:
    """Search the web for today's weather conditions, forecasts,
    weather alerts, and breaking weather Snews for the specified city.
    Use this tool for current weather information, not general trivia.
    """
    today = date.today().strftime("%B %d, %Y")
    query = f"weather news forecast alerts {city} {today}"

    try:
        results = DDGS().text(query, max_results=5)

        if not results:
            return f"No current weather news found for {city}."

        return "\n".join(
            f"Title: {item.get('title', '')}\n"
            f"Details: {item.get('body', '')}\n"
            f"Source: {item.get('href', '')}"
            for item in results
        )

    except Exception as e:
        return f"Weather search failed: {e}"


async def main():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing from .env")

    # 2. Create the OpenAI model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key,
    )

    try:
        # 3. Create the AutoGen agent with the search tool
        weather_agent = AssistantAgent(
            name="weather_news_agent",
            model_client=model_client,
            tools=[search_weather_news],
            reflect_on_tool_use=True,
            system_message=(
                "You are a Weather News assistant. "
                "Use the search_weather_news tool to find current "
                "weather news for the user's city. Base your answer "
                "on search results; do not invent weather conditions. "
                "Reply in exactly 3 short lines: "
                "1. Current weather or temperature, if available. "
                "2. Forecast or important weather alerts. "
                "3. A useful precaution or a brief source note. "
                "If information is unavailable, say so clearly. "
                "Do not add headings or extra lines."
            ),
        )

        # 4. Ask the user for a city
        city = input("Enter your city: ").strip()

        if not city:
            print("Please enter a valid city.")
            return

        # 5. Run the agent
        result = await weather_agent.run(
            task=f"Find today's weather news for {city}."
        )

        # Print the final assistant response
        for message in reversed(result.messages):
            if (
                getattr(message, "source", None)
                == "weather_news_agent"
                and isinstance(getattr(message, "content", None), str)
            ):
                print(message.content)
                break

    finally:
        # 6. Close the model client
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
