from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from typing import TypedDict, Annotated, List, Union
import operator
import os
from dotenv import load_dotenv
import json
import yfinance as yf
# import pandas as pd


load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_KEY_API")
if not GROQ_API_KEY:
    raise ValueError("GROQ_KEY_API not found in environment variables. Please set it in a .env file.")

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    groq_api_key=GROQ_API_KEY,
)



class FinancialMetrics(BaseModel):
    """Data model for the final financial analysis output."""
    symbol: str = Field(description="Stock ticker symbol")
    current_price: float = Field(description="Current market price")
    moving_average: float = Field(description="Moving average value")
    recommendation: str = Field(description="Buy/Hold/Sell recommendation")


class FinancialAnalysisRequest(BaseModel):
    """Data model for parsing the user's request."""
    symbol: str = Field(description="Stock ticker symbol to analyze")
    window: int = Field(description="Time window for moving average, defaulting to 14 if not specified.")



@tool
def get_current_price(symbol: str) -> float:
    """
    Fetch the most recent stock price for a given symbol using the yfinance library.
    """
    symbol = symbol.upper()
    ticker = yf.Ticker(symbol)

    # Fetch daily data; the last 'Close' is a good proxy for the current price
    hist = ticker.history(period="1d")
    if hist.empty:
        raise ValueError(f"Invalid or unknown symbol: '{symbol}'. No data found.")

    return round(hist['Close'].iloc[-1], 2)


@tool
def calculate_moving_average(symbol: str, window: int) -> float:
    """
    Calculate the moving average for a given stock symbol over a specified window
    by fetching historical data from yfinance.
    """
    symbol = symbol.upper()
    if not isinstance(window, int) or window <= 0:
        raise ValueError("Window size must be a positive integer.")

    ticker = yf.Ticker(symbol)


    hist = ticker.history(period="1y")


    if hist.empty:
        raise ValueError(f"Invalid or unknown symbol: '{symbol}'. No historical data found.")

    if len(hist) < window:
        raise ValueError(
            f"Not enough historical data for symbol '{symbol}' to calculate a {window}-day moving average. Only {len(hist)} days available.")

    # Calculate the simple moving average on the 'Close' price
    moving_avg = hist['Close'].rolling(window=window).mean().iloc[-1]

    return round(moving_avg, 2)



class AgentState(TypedDict):
    """Represents the state of our agent, passed between nodes."""
    input: Union[str, dict]
    intermediate_steps: Annotated[List[tuple[dict, str]], operator.add]
    output: Union[FinancialMetrics, dict, None]



def parse_input(state: AgentState) -> dict:
    """
    Parses the initial input, which can be a string or a dictionary.
    Uses an LLM to convert natural language into a structured request.
    """
    input_data = state["input"]

    if isinstance(input_data, dict) and "symbol" in input_data and "window" in input_data:
        return {"input": input_data, "intermediate_steps": []}

    parser = JsonOutputParser(pydantic_object=FinancialAnalysisRequest)

    prompt = ChatPromptTemplate.from_template(
        "You are an expert at parsing financial queries. Convert the user's request "
        "into a JSON object with 'symbol' and 'window' keys. The symbol should be a valid stock ticker. "
        "If the user does not specify a window, use a default of 14.\n\n"
        "QUERY: {query}\n\n"
        "JSON_OUTPUT:"
    )

    try:
        query_text = str(input_data)
        chain = prompt | llm | parser
        parsed_request = chain.invoke({"query": query_text})
        return {"input": parsed_request, "intermediate_steps": []}
    except Exception as e:
        error_message = f"Failed to parse input. Error: {e}. Please provide a clear stock symbol and optional window."
        return {"output": {"error": error_message}}


def analyze_financials(state: AgentState) -> dict:
    """
    Runs the financial analysis by calling the defined tools with live data.
    """
    if "error" in state.get("output", {}):
        return {}

    try:
        symbol = state["input"]["symbol"]
        window = state["input"]["window"]

        current_price = get_current_price.invoke({"symbol": symbol})
        moving_avg = calculate_moving_average.invoke({"symbol": symbol, "window": window})

        if current_price < moving_avg:
            recommendation = "Buy"
        elif current_price > moving_avg:
            recommendation = "Sell"
        else:
            recommendation = "Hold"

        return {
            "output": FinancialMetrics(
                symbol=symbol,
                current_price=current_price,
                moving_average=moving_avg,
                recommendation=recommendation
            )
        }
    except Exception as e:
        return {"output": {"error": str(e)}}


def format_output(state: AgentState) -> dict:
    """Final node to prepare the output."""
    return {"output": state["output"]}



workflow = StateGraph(AgentState)

workflow.add_node("input_parser", parse_input)
workflow.add_node("financial_analyzer", analyze_financials)
workflow.add_node("output_formatter", format_output)

workflow.set_entry_point("input_parser")
workflow.add_edge("input_parser", "financial_analyzer")
workflow.add_edge("financial_analyzer", "output_formatter")
workflow.add_edge("output_formatter", END)

financial_agent = workflow.compile()


if __name__ == "__main__":
    test_cases = [
        "What is the analysis for Tesla with a 50-day moving average?",  # Natural language
        {"symbol": "ATAI", "window": 20},
        "GOOGL moving average",
        "Give me the numbers for Pfizer",
        "Give me the numbers for TSM",
        "Give me the numbers for HD",
    ]

    for i, test_input in enumerate(test_cases):
        print(f"\n{'=' * 40}\nTEST CASE {i + 1}: {test_input}\n{'=' * 40}")

        try:
            result = financial_agent.invoke({"input": test_input})

            final_output = result.get("output")
            if isinstance(final_output, BaseModel):
                print(json.dumps(final_output.dict(), indent=2))
            else:
                print(json.dumps(final_output, indent=2))

        except Exception as e:
            print(f"An unexpected error occurred during agent execution: {e}")
