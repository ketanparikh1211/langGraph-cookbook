"""
Cricket Statistics Workflow using LangGraph

This script demonstrates parallel processing in LangGraph by calculating 
multiple cricket statistics simultaneously before generating a summary.

Key concepts:
- Parallel execution of nodes
- Fan-out/fan-in pattern for concurrent processing
- Aggregating results from parallel operations
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Define the state structure for cricket batsman statistics
class BatsmanState(TypedDict):
    """State definition for cricket batsman statistics workflow.
    
    Attributes:
        runs: Total runs scored by the batsman
        balls: Total balls faced by the batsman
        fours: Number of boundaries (4 runs) hit
        sixes: Number of sixes (6 runs) hit
        sr: Strike rate (runs per 100 balls)
        bpb: Balls per boundary ratio
        boundary_percent: Percentage of runs from boundaries
        summary: Text summary of all statistics
    """
    runs: int
    balls: int
    fours: int
    sixes: int

    sr: float
    bpb: float
    boundary_percent: float
    summary: str

def calculate_sr(state: BatsmanState):
    """Calculate strike rate (runs per 100 balls).
    
    Formula: (runs / balls) * 100
    
    Args:
        state: Current workflow state with runs and balls
        
    Returns:
        Dictionary with calculated strike rate
    """
    sr = (state['runs']/state['balls'])*100
    
    return {'sr': sr}

def calculate_bpb(state: BatsmanState):
    """Calculate balls per boundary ratio.
    
    Formula: balls / (fours + sixes)
    
    Args:
        state: Current workflow state with balls, fours, and sixes
        
    Returns:
        Dictionary with calculated balls per boundary
    """
    bpb = state['balls']/(state['fours'] + state['sixes'])

    return {'bpb': bpb}

def calculate_boundary_percent(state: BatsmanState):
    """Calculate percentage of runs scored from boundaries.
    
    Formula: ((fours * 4) + (sixes * 6)) / runs * 100
    
    Args:
        state: Current workflow state with runs, fours, and sixes
        
    Returns:
        Dictionary with calculated boundary percentage
    """
    boundary_percent = (((state['fours'] * 4) + (state['sixes'] * 6))/state['runs'])*100

    return {'boundary_percent': boundary_percent}

def summary(state: BatsmanState):
    """Generate a text summary of all calculated statistics.
    
    Args:
        state: Current workflow state with all calculated statistics
        
    Returns:
        Dictionary with formatted summary text
    """
    summary = f"""
Strike Rate - {state['sr']} \n
Balls per boundary - {state['bpb']} \n
Boundary percent - {state['boundary_percent']}
"""
    
    return {'summary': summary}

# Create the workflow graph
graph = StateGraph(BatsmanState)

# Add nodes for each calculation
graph.add_node('calculate_sr', calculate_sr)
graph.add_node('calculate_bpb', calculate_bpb)
graph.add_node('calculate_boundary_percent', calculate_boundary_percent)
graph.add_node('summary', summary)

# Define parallel flow:
# START -> [calculate_sr, calculate_bpb, calculate_boundary_percent] -> summary -> END
# This creates a fan-out/fan-in pattern for parallel processing
graph.add_edge(START, 'calculate_sr')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_boundary_percent')

graph.add_edge('calculate_sr', 'summary')
graph.add_edge('calculate_bpb', 'summary')
graph.add_edge('calculate_boundary_percent', 'summary')

graph.add_edge('summary', END)

# Compile the workflow
workflow = graph.compile()

# Execute with initial statistics
initial_state = {
    'runs': 100,
    'balls': 50,
    'fours': 6,
    'sixes': 4
}

# Run the workflow and print results
final_state = workflow.invoke(initial_state)
print("Batsman Statistics:")
print(f"Runs: {final_state['runs']}")
print(f"Balls: {final_state['balls']}")
print(f"Strike Rate: {final_state['sr']:.2f}")
print(f"Balls per Boundary: {final_state['bpb']:.2f}")
print(f"Boundary Percentage: {final_state['boundary_percent']:.2f}%")
print("\nSummary:")
print(final_state['summary'])