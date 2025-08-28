"""
Quadratic Equation Solver using LangGraph

This script demonstrates conditional branching in LangGraph by implementing
a solver for quadratic equations that takes different paths based on the
value of the discriminant.

Key concepts:
- Conditional routing with branching logic
- Using Literal types for type-safe branch selection
- Handling multiple potential outcomes in a workflow
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal

# Define the state structure for quadratic equation solving
class QuadState(TypedDict):
    """State definition for quadratic equation solver workflow.
    
    Attributes:
        a: Coefficient of x²
        b: Coefficient of x
        c: Constant term
        equation: String representation of the equation
        discriminant: Value of b² - 4ac
        result: Text description of the solution
    """
    a: int
    b: int
    c: int

    equation: str
    discriminant: float
    result: str

def show_equation(state: QuadState):
    """Format the quadratic equation as a string.
    
    Args:
        state: Current workflow state with coefficients a, b, c
        
    Returns:
        Dictionary with formatted equation string
    """
    equation = f'{state["a"]}x² + {state["b"]}x + {state["c"]}'

    return {'equation': equation}

def calculate_discriminant(state: QuadState):
    """Calculate the discriminant of the quadratic equation.
    
    Formula: b² - 4ac
    
    Args:
        state: Current workflow state with coefficients a, b, c
        
    Returns:
        Dictionary with calculated discriminant value
    """
    discriminant = state["b"]**2 - (4*state["a"]*state["c"])

    return {'discriminant': discriminant}

def real_roots(state: QuadState):
    """Calculate two distinct real roots when discriminant > 0.
    
    Formula: (-b ± √discriminant) / 2a
    
    Args:
        state: Current workflow state with coefficients and discriminant
        
    Returns:
        Dictionary with result string containing both roots
    """
    root1 = (-state["b"] + state["discriminant"]**0.5)/(2*state["a"])
    root2 = (-state["b"] - state["discriminant"]**0.5)/(2*state["a"])

    result = f'The roots are {root1} and {root2}'

    return {'result': result}

def repeated_roots(state: QuadState):
    """Calculate repeated root when discriminant = 0.
    
    Formula: -b / 2a
    
    Args:
        state: Current workflow state with coefficients
        
    Returns:
        Dictionary with result string containing the repeated root
    """
    root = (-state["b"])/(2*state["a"])

    result = f'Only one repeated root exists: {root}'

    return {'result': result}

def no_real_roots(state: QuadState):
    """Handle case when discriminant < 0 (no real roots).
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with result indicating no real roots
    """
    result = f'No real roots exist (discriminant < 0)'

    return {'result': result}

def check_condition(state: QuadState) -> Literal["real_roots", "repeated_roots", "no_real_roots"]:
    """Determine which branch to take based on discriminant value.
    
    Args:
        state: Current workflow state with discriminant
        
    Returns:
        String literal indicating which branch to follow
    """
    if state['discriminant'] > 0:
        return "real_roots"
    elif state['discriminant'] == 0:
        return "repeated_roots"
    else:
        return "no_real_roots"

# Create the workflow graph
graph = StateGraph(QuadState)

# Add nodes for each step in the workflow
graph.add_node('show_equation', show_equation)
graph.add_node('calculate_discriminant', calculate_discriminant)
graph.add_node('real_roots', real_roots)
graph.add_node('repeated_roots', repeated_roots)
graph.add_node('no_real_roots', no_real_roots)

# Define the linear part of the flow
graph.add_edge(START, 'show_equation')
graph.add_edge('show_equation', 'calculate_discriminant')

# Define conditional branches based on discriminant value
# This is where the path splits into three possibilities
graph.add_conditional_edges(
    'calculate_discriminant',   # Source node
    check_condition,            # Function that determines the branch
    {                           # Mapping of return values to target nodes
        "real_roots": 'real_roots',
        "repeated_roots": 'repeated_roots',
        "no_real_roots": 'no_real_roots'
    }
)

# All branches end the workflow
graph.add_edge('real_roots', END)
graph.add_edge('repeated_roots', END)
graph.add_edge('no_real_roots', END)

# Compile the workflow
workflow = graph.compile()

# Example inputs
initial_state = {
    'a': 2, 
    'b': 4,
    'c': 2
}

# Run the workflow and print results
final_state = workflow.invoke(initial_state)
print(f"Equation: {final_state['equation']}")
print(f"Discriminant: {final_state['discriminant']}")
print(f"Result: {final_state['result']}")
