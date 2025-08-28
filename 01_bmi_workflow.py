"""
BMI Calculator Workflow using LangGraph

This script demonstrates a basic workflow using LangGraph's StateGraph
to calculate BMI (Body Mass Index) and categorize it.

Key concepts:
- Creating a simple StateGraph
- Defining state using TypedDict
- Creating computational nodes for each step
- Connecting nodes with edges
- Compiling and invoking the graph
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from IPython.display import Image

# Define the state structure using TypedDict
# This enforces type checking and documents the expected structure
class BMIState(TypedDict):
    """State definition for BMI calculation workflow.
    
    Attributes:
        weight_kg: Weight in kilograms
        height_m: Height in meters
        bmi: Calculated BMI value
        category: BMI category classification
    """
    weight_kg: float
    height_m: float
    bmi: float
    category: str

def calculate_bmi(state: BMIState) -> BMIState:
    """Calculate BMI from weight and height.
    
    BMI formula: weight (kg) / height^2 (m)
    
    Args:
        state: Current workflow state containing weight and height
        
    Returns:
        Updated state with BMI value
    """
    weight = state['weight_kg']
    height = state['height_m']

    bmi = weight/(height**2)

    # Round to 2 decimal places for readability
    state['bmi'] = round(bmi, 2)

    return state

def label_bmi(state: BMIState) -> BMIState:
    """Categorize BMI value according to standard ranges.
    
    BMI Categories:
    - Under 18.5: Underweight
    - 18.5 to 24.9: Normal
    - 25 to 29.9: Overweight
    - 30 and above: Obese
    
    Args:
        state: Current workflow state containing BMI value
        
    Returns:
        Updated state with BMI category
    """
    bmi = state['bmi']

    if bmi < 18.5:
        state["category"] = "Underweight"
    elif 18.5 <= bmi < 25:
        state["category"] = "Normal"
    elif 25 <= bmi < 30:
        state["category"] = "Overweight"
    else:
        state["category"] = "Obese"

    return state

# Create a workflow graph
# StateGraph is the core structure that defines our workflow
graph = StateGraph(BMIState)

# Add nodes to the graph
# Each node is a function that processes the state
graph.add_node('calculate_bmi', calculate_bmi)
graph.add_node('label_bmi', label_bmi)

# Define the flow between nodes using edges
# START and END are special markers for the entry and exit points
graph.add_edge(START, 'calculate_bmi')  # Start with BMI calculation
graph.add_edge('calculate_bmi', 'label_bmi')  # Then categorize the result
graph.add_edge('label_bmi', END)  # End workflow after categorization

# Compile the graph into an executable workflow
workflow = graph.compile()

# Execute the workflow with initial input values
initial_state = {'weight_kg': 80, 'height_m': 1.73}
final_state = workflow.invoke(initial_state)

print(final_state)

# Visualization of the workflow (only works in Jupyter/IPython environments)
Image(workflow.get_graph().draw_mermaid_png())