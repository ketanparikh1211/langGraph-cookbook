"""
Blog Generation Workflow using LangGraph

This script demonstrates prompt chaining by creating a blog post generation workflow
with sequential LLM calls - first creating an outline, then using that outline to write
a full blog post.

Key concepts:
- Sequential chaining of LLM calls
- Building on previous outputs in a workflow
- Using more complex state management
"""

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv
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
    except:
        print("Could not determine environment file path")
else:
    print("No API key found in environment variables!")
    print("Please ensure you have an OPENAI_API_KEY in your .env file")
    exit(1)  # Exit if no API key is found

# Initialize the language model
model = ChatOpenAI(openai_api_key=api_key)

# Define the state structure for our blog creation workflow
class BlogState(TypedDict):
    """State definition for blog generation workflow.
    
    Attributes:
        title: The blog post title
        outline: The generated outline for the blog
        content: The full generated blog content
    """
    title: str
    outline: str
    content: str

def create_outline(state: BlogState) -> BlogState:
    """Generate an outline for a blog post based on the given title.
    
    Args:
        state: Current workflow state containing the blog title
        
    Returns:
        Updated state with the generated outline
    """
    # Fetch title from state
    title = state['title']

    # Call LLM to generate outline
    prompt = f'Generate a detailed outline for a blog on the topic - {title}'
    outline = model.invoke(prompt).content

    # Update state with the generated outline
    state['outline'] = outline

    return state

def create_blog(state: BlogState) -> BlogState:
    """Generate a full blog post using the title and outline.
    
    This function builds on the output of the create_outline function,
    demonstrating how to chain LLM calls in a workflow.
    
    Args:
        state: Current workflow state containing title and outline
        
    Returns:
        Updated state with the generated blog content
    """
    title = state['title']
    outline = state['outline']

    # Create a prompt that includes both the title and outline
    prompt = f'Write a detailed blog on the title - {title} using the following outline:\n{outline}'

    # Generate the blog content
    content = model.invoke(prompt).content

    # Update state with the generated content
    state['content'] = content

    return state

# Create the workflow graph
graph = StateGraph(BlogState)

# Add nodes to the graph
graph.add_node('create_outline', create_outline)
graph.add_node('create_blog', create_blog)

# Define the sequential flow:
# START -> create_outline -> create_blog -> END
graph.add_edge(START, 'create_outline')
graph.add_edge('create_outline', 'create_blog')
graph.add_edge('create_blog', END)

# Compile the workflow
workflow = graph.compile()

# Execute with an initial blog title
initial_state = {'title': 'Rise of AI in India'}
final_state = workflow.invoke(initial_state)

# Print the resulting blog
print("--- Blog Title ---")
print(final_state['title'])
print("\n--- Blog Outline ---")
print(final_state['outline'])
print("\n--- Blog Content ---")
print(final_state['content'])