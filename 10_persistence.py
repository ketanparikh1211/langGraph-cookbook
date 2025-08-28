"""
Joke Generator with Persistence using LangGraph

This script demonstrates state persistence in LangGraph by creating a joke generation
workflow that can save and retrieve state across multiple sessions or threads.

Key concepts:
- Using checkpointers for state persistence
- Managing multiple conversation threads
- Retrieving and updating persisted state
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
import os

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

# Define the state structure for our joke workflow
class JokeState(TypedDict):
    """State definition for joke generation workflow.
    
    Attributes:
        topic: The subject for the joke
        joke: The generated joke text
        explanation: An explanation of the humor in the joke
    """
    topic: str
    joke: str
    explanation: str

# Define node functions for the workflow
def generate_joke(state: JokeState):
    """Generate a joke based on the provided topic.
    
    Args:
        state: Current workflow state containing the topic
        
    Returns:
        Dictionary with generated joke
    """
    prompt = f'Generate a short, funny joke on the topic: {state["topic"]}'
    response = llm.invoke(prompt).content
    return {'joke': response}

def generate_explanation(state: JokeState):
    """Generate an explanation for the joke.
    
    Args:
        state: Current workflow state containing the joke
        
    Returns:
        Dictionary with joke explanation
    """
    prompt = f'Write a brief explanation for this joke: {state["joke"]}'
    response = llm.invoke(prompt).content
    return {'explanation': response}

# Create and configure the workflow
def create_joke_workflow():
    """Create and return a compiled workflow for joke generation with persistence.
    
    Returns:
        A compiled StateGraph with persistence enabled
    """
    # Create the graph with our state type
    graph = StateGraph(JokeState)
    
    # Add nodes for each step
    graph.add_node('generate_joke', generate_joke)
    graph.add_node('generate_explanation', generate_explanation)
    
    # Define the sequential flow
    graph.add_edge(START, 'generate_joke')
    graph.add_edge('generate_joke', 'generate_explanation')
    graph.add_edge('generate_explanation', END)
    
    # Set up persistence with InMemorySaver
    # This stores state in memory, but could be replaced with a database-backed solution
    checkpointer = InMemorySaver()
    
    # Compile and return the workflow with persistence enabled
    return graph.compile(checkpointer=checkpointer)

def main():
    """Main function to demonstrate LangGraph persistence capabilities."""
    # Create the workflow
    workflow = create_joke_workflow()
    
    print("\n--- Demonstrating LangGraph Persistence ---\n")
    
    # Example 1: Create a joke about pizza with thread_id 1
    config1 = {"configurable": {"thread_id": "1"}}
    print("Running workflow with topic 'pizza' (thread_id: 1)")
    result1 = workflow.invoke({'topic': 'pizza'}, config=config1)
    print("\nResult 1:")
    print(f"Topic: pizza")
    print(f"Joke: {result1['joke']}")
    print(f"Explanation: {result1['explanation']}")
    
    # Retrieve the current state for thread 1
    print("\nCurrent state for thread_id 1:")
    state1 = workflow.get_state(config1)
    print(state1)
    
    # Retrieve the state history for thread 1
    print("\nState history for thread_id 1:")
    history1 = list(workflow.get_state_history(config1))
    for i, entry in enumerate(history1):
        print(f"History entry {i+1}: {entry}")
    
    # Example 2: Create a joke about pasta with thread_id 2
    # This demonstrates managing multiple conversation threads
    config2 = {"configurable": {"thread_id": "2"}}
    print("\nRunning workflow with topic 'pasta' (thread_id: 2)")
    result2 = workflow.invoke({'topic': 'pasta'}, config=config2)
    print("\nResult 2:")
    print(f"Topic: pasta")
    print(f"Joke: {result2['joke']}")
    print(f"Explanation: {result2['explanation']}")
    
    # Verify that thread 1's state is still available
    # This demonstrates that separate threads maintain separate states
    print("\nVerifying thread_id 1's state is still available:")
    state1_again = workflow.get_state(config1)
    print(state1_again)
    
    # Example 3: Update the state of thread 1 to change the topic to 'samosa'
    # This demonstrates modifying persisted state
    print("\nUpdating thread_id 1's state to change topic to 'samosa'")
    # Access the checkpoint_id from the config attribute of the StateSnapshot object
    checkpoint_id = history1[0].config['configurable']['checkpoint_id']
    update_config = {
        "configurable": {
            "thread_id": "1",
            "checkpoint_id": checkpoint_id,
            "checkpoint_ns": ""
        }
    }
    
    # Update the state with the new topic
    updated_state = workflow.update_state(update_config, {'topic': 'samosa'})
    print(f"Updated state config: {updated_state}")
    
    # Example 4: Continue the workflow from the updated state
    # This demonstrates resuming execution with modified state
    print("\nContinuing workflow from updated state (topic: samosa)")
    result3 = workflow.invoke(None, {"configurable": {"thread_id": "1", "checkpoint_id": updated_state['configurable']['checkpoint_id']}})
    print("\nResult 3:")
    print(f"Topic: samosa")
    print(f"Joke: {result3['joke']}")
    print(f"Explanation: {result3['explanation']}")
    
    # Show the complete history after all operations
    print("\nFinal state history for thread_id 1:")
    final_history = list(workflow.get_state_history(config1))
    for i, entry in enumerate(final_history):
        print(f"History entry {i+1}: {entry}")

# Run the demonstration if this script is executed directly
if __name__ == "__main__":
    main()
