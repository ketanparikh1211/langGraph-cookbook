"""
LangGraph with Tools Integration

This script demonstrates how to incorporate external tools (search, calculator, stock price API)
into a LangGraph workflow to enhance the capabilities of language models.

Key concepts:
- Integrating external tools with LLMs
- Using LangGraph's ToolNode
- Implementing conditional routing for tool usage
- Building a conversational agent with tool access
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

import requests
import random

# Load environment variables from .env file
load_dotenv(override=True)

# Validate API key configuration
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    print(f"API key found! First 5 chars: {api_key[:5]}...")
    print(f"API key length: {len(api_key)}")
    # Print the key source if possible
    try:
        from dotenv import find_dotenv
        env_path = find_dotenv()
        print(f"Environment file path: {env_path}")
    except Exception as e:
        print(f"Could not determine environment file path: {e}")
else:
    print("No API key found in environment variables!")
    print("Please ensure you have an OPENAI_API_KEY in your .env file")
    exit(1)  # Exit if no API key is found

# Initialize the language model
llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=api_key)

# Define available tools
# 1. Web search tool using DuckDuckGo
search_tool = DuckDuckGoSearchRun(region="us-en")

# 2. Custom calculator tool with decorator
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    
    Args:
        first_num: First number in the operation
        second_num: Second number in the operation
        operation: Type of operation to perform (add, sub, mul, div)
    
    Returns:
        Dictionary with operation details and result
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {
            "first_num": first_num, 
            "second_num": second_num, 
            "operation": operation, 
            "result": result
        }
    except Exception as e:
        return {"error": str(e)}

# 3. Custom stock price tool
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
    
    Returns:
        Dictionary with stock price information from Alpha Vantage API
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

# Compile the tools into a list
tools = [get_stock_price, search_tool, calculator]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)

# Define the state structure for our chatbot
class ChatState(TypedDict):
    """State definition for tool-enabled chatbot.
    
    Attributes:
        messages: Conversation history with special annotation for handling message lists
    """
    messages: Annotated[list[BaseMessage], add_messages]

# Define the main chat node that may use tools
def chat_node(state: ChatState):
    """Process the conversation and either respond directly or request a tool call.
    
    Args:
        state: Current workflow state with message history
        
    Returns:
        Dictionary with LLM response to be added to messages
    """
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Create a ToolNode that handles tool execution
tool_node = ToolNode(tools)  # Executes tool calls when requested by the LLM

# Create the workflow graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")

# Add conditional branching for tool usage
# If the LLM requested a tool, go to ToolNode; otherwise end the conversation
graph.add_conditional_edges(
    "chat_node",
    tools_condition  # Special function that checks if LLM requested a tool
)

# Complete the cycle by returning to chat_node after tool execution
graph.add_edge("tools", "chat_node")    

# Compile the workflow
chatbot = graph.compile()

# Example 1: Regular chat without tools
print("\n--- Example 1: Regular Chat ---")
out = chatbot.invoke({"messages": [HumanMessage(content="Hello!")]})
print(f"User: Hello!")
print(f"AI: {out['messages'][-1].content}")

# Example 2: Chat requiring calculator tool
print("\n--- Example 2: Using Calculator ---")
out = chatbot.invoke({"messages": [HumanMessage(content="What is 2*3?")]})
print(f"User: What is 2*3?")
print(f"AI: {out['messages'][-1].content}")

# Example 3: Chat requiring stock price tool
print("\n--- Example 3: Using Stock Price Tool ---")
out = chatbot.invoke({"messages": [HumanMessage(content="What is the stock price of Apple?")]})
print(f"User: What is the stock price of Apple?")
print(f"AI: {out['messages'][-1].content}")

# Example 4: Complex query using multiple tools
print("\n--- Example 4: Multi-tool Query ---")
out = chatbot.invoke({"messages": [HumanMessage(content="First find out the stock price of Apple using get stock price tool then use the calculator tool to find out how much will it take to purchase 50 shares?")]})
print(f"User: First find out the stock price of Apple using get stock price tool then use the calculator tool to find out how much will it take to purchase 50 shares?")
print(f"AI: {out['messages'][-1].content}")