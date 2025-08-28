"""
Simple Chatbot using LangGraph

This script demonstrates message-based state management in LangGraph
for creating a basic chatbot with memory of the conversation history.

Key concepts:
- Message-based state management
- Using the add_messages annotation for proper message handling
- Creating a stateful conversational agent
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
import operator
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os
from langgraph.graph.message import add_messages

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
    except:
        print("Could not determine environment file path")
else:
    print("No API key found in environment variables!")
    print("Please ensure you have an OPENAI_API_KEY in your .env file")
    exit(1)  # Exit if no API key is found

# Define the state structure for our chatbot
class ChatState(TypedDict):
    """State definition for chatbot workflow.
    
    Attributes:
        messages: List of conversation messages with special annotation for handling message lists
    """
    messages: Annotated[list[BaseMessage], add_messages]

# Initialize the language model
llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=api_key)

def chat_node(state: ChatState):
    """Process the conversation history and generate a response.
    
    Args:
        state: Current workflow state containing the message history
        
    Returns:
        Dictionary with LLM response to be added to messages
    """
    # Extract the conversation history
    messages = state['messages']
    
    # Generate response from LLM
    response = llm.invoke(messages)
    
    # Return response to be added to message history
    # The add_messages annotation will handle proper concatenation
    return {'messages': [response]}

# Create the workflow graph
graph = StateGraph(ChatState)

# Add the chat processing node
graph.add_node('chat_node', chat_node)

# Define the simplest possible flow:
# START -> chat_node -> END
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# Compile the workflow
chatbot = graph.compile()

# Execute with an initial query
initial_state = {
    'messages': [HumanMessage(content='What is the capital of India?')]
}

# Run the workflow and print the response
response = chatbot.invoke(initial_state)
print("User: What is the capital of India?")
print(f"Chatbot: {response['messages'][-1].content}")

# You can continue the conversation by providing the updated state
# with the new message appended to the history
follow_up = {
    'messages': response['messages'] + [HumanMessage(content='Tell me more about its history.')]
}

# This would maintain conversation context
follow_up_response = chatbot.invoke(follow_up)
print("\nUser: Tell me more about its history.")
print(f"Chatbot: {follow_up_response['messages'][-1].content}")