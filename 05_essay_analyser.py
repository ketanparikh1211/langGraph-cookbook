"""
Essay Analyzer Workflow using LangGraph

This script demonstrates structured output and parallel aggregation in LangGraph
by analyzing an essay from multiple perspectives and aggregating the results.

Key concepts:
- Using Pydantic models for structured LLM outputs
- Parallel processing with result aggregation
- Operator-based list aggregation using operator.add
"""

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator
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
model = ChatOpenAI(model='gpt-4o-mini', openai_api_key=api_key)

# Define Pydantic schema for structured output
class EvaluationSchema(BaseModel):
    """Schema for essay evaluation output.
    
    Attributes:
        feedback: Detailed textual feedback
        score: Numerical score from 0-10
    """
    feedback: str = Field(description='Detailed feedback for the essay')
    score: int = Field(description='Score out of 10', ge=0, le=10)

# Create a model that outputs structured data according to our schema
structured_model = model.with_structured_output(EvaluationSchema)

# Sample essay for demonstration
essay = """India in the Age of AI
As the world enters a transformative era defined by artificial intelligence (AI), India stands at a critical juncture — one where it can either emerge as a global leader in AI innovation or risk falling behind in the technology race. The age of AI brings with it immense promise as well as unprecedented challenges, and how India navigates this landscape will shape its socio-economic and geopolitical future.

India's strengths in the AI domain are rooted in its vast pool of skilled engineers, a thriving IT industry, and a growing startup ecosystem. With over 5 million STEM graduates annually and a burgeoning base of AI researchers, India possesses the intellectual capital required to build cutting-edge AI systems. Institutions like IITs, IIITs, and IISc have begun fostering AI research, while private players such as TCS, Infosys, and Wipro are integrating AI into their global services. In 2020, the government launched the National AI Strategy (AI for All) with a focus on inclusive growth, aiming to leverage AI in healthcare, agriculture, education, and smart mobility.

One of the most promising applications of AI in India lies in agriculture, where predictive analytics can guide farmers on optimal sowing times, weather forecasts, and pest control. In healthcare, AI-powered diagnostics can help address India's doctor-patient ratio crisis, particularly in rural areas. Educational platforms are increasingly using AI to personalize learning paths, while smart governance tools are helping improve public service delivery and fraud detection.

However, the path to AI-led growth is riddled with challenges. Chief among them is the digital divide. While metropolitan cities may embrace AI-driven solutions, rural India continues to struggle with basic internet access and digital literacy. The risk of job displacement due to automation also looms large, especially for low-skilled workers. Without effective skilling and re-skilling programs, AI could exacerbate existing socio-economic inequalities.

Another pressing concern is data privacy and ethics. As AI systems rely heavily on vast datasets, ensuring that personal data is used transparently and responsibly becomes vital. India is still shaping its data protection laws, and in the absence of a strong regulatory framework, AI systems may risk misuse or bias.

To harness AI responsibly, India must adopt a multi-stakeholder approach involving the government, academia, industry, and civil society. Policies should promote open datasets, encourage responsible innovation, and ensure ethical AI practices. There is also a need for international collaboration, particularly with countries leading in AI research, to gain strategic advantage and ensure interoperability in global systems.

India's demographic dividend, when paired with responsible AI adoption, can unlock massive economic growth, improve governance, and uplift marginalized communities. But this vision will only materialize if AI is seen not merely as a tool for automation, but as an enabler of human-centered development.

In conclusion, India in the age of AI is a story in the making — one of opportunity, responsibility, and transformation. The decisions we make today will not just determine India's AI trajectory, but also its future as an inclusive, equitable, and innovation-driven society."""

# Define the state structure for UPSC essay analysis
class UPSCState(TypedDict):
    """State definition for UPSC essay analysis workflow.
    
    Attributes:
        essay: The essay text to analyze
        language_feedback: Feedback on language quality
        analysis_feedback: Feedback on depth of analysis
        clarity_feedback: Feedback on clarity of thought
        overall_feedback: Aggregated final feedback
        individual_scores: List of individual scores (aggregated with operator.add)
        avg_score: Average score calculated from individual scores
    """
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]  # Special annotation for list aggregation
    avg_score: float

def evaluate_language(state: UPSCState):
    """Evaluate the language quality of the essay.
    
    Args:
        state: Current workflow state containing the essay
        
    Returns:
        Dictionary with language feedback and score
    """
    prompt = f'Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
    output = structured_model.invoke(prompt)

    return {'language_feedback': output.feedback, 'individual_scores': [output.score]}

def evaluate_analysis(state: UPSCState):
    """Evaluate the depth of analysis in the essay.
    
    Args:
        state: Current workflow state containing the essay
        
    Returns:
        Dictionary with analysis feedback and score
    """
    prompt = f'Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
    output = structured_model.invoke(prompt)

    return {'analysis_feedback': output.feedback, 'individual_scores': [output.score]}

def evaluate_thought(state: UPSCState):
    """Evaluate the clarity of thought in the essay.
    
    Args:
        state: Current workflow state containing the essay
        
    Returns:
        Dictionary with clarity feedback and score
    """
    prompt = f'Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
    output = structured_model.invoke(prompt)

    return {'clarity_feedback': output.feedback, 'individual_scores': [output.score]}

def final_evaluation(state: UPSCState):
    """Generate a final evaluation by aggregating all feedback.
    
    Args:
        state: Current workflow state containing all feedback
        
    Returns:
        Dictionary with overall feedback and average score
    """
    # Create a summary feedback based on all individual feedback
    prompt = f"""Based on the following feedbacks create a summarized feedback:
    
Language feedback: {state["language_feedback"]}
Depth of analysis feedback: {state["analysis_feedback"]}
Clarity of thought feedback: {state["clarity_feedback"]}
"""
    overall_feedback = model.invoke(prompt).content

    # Calculate average score from individual scores
    avg_score = sum(state['individual_scores'])/len(state['individual_scores'])

    return {'overall_feedback': overall_feedback, 'avg_score': avg_score}

# Create the workflow graph
graph = StateGraph(UPSCState)

# Add nodes for each evaluation aspect
graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_thought', evaluate_thought)
graph.add_node('final_evaluation', final_evaluation)

# Define parallel flow for the evaluations
# START -> [evaluate_language, evaluate_analysis, evaluate_thought] -> final_evaluation -> END
graph.add_edge(START, 'evaluate_language')
graph.add_edge(START, 'evaluate_analysis')
graph.add_edge(START, 'evaluate_thought')

graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_thought', 'final_evaluation')

graph.add_edge('final_evaluation', END)

# Compile the workflow
workflow = graph.compile()

# Example essay for evaluation
essay2 = """India and AI Time

Now world change very fast because new tech call Artificial Intel… something (AI). India also want become big in this AI thing. If work hard, India can go top. But if no careful, India go back.

India have many good. We have smart student, many engine-ear, and good IT peoples. Big company like TCS, Infosys, Wipro already use AI. Government also do program "AI for All". It want AI in farm, doctor place, school and transport.

In farm, AI help farmer know when to put seed, when rain come, how stop bug. In health, AI help doctor see sick early. In school, AI help student learn good. Government office use AI to find bad people and work fast.

But problem come also. First is many villager no have phone or internet. So AI not help them. Second, many people lose job because AI and machine do work. Poor people get more bad.

One more big problem is privacy. AI need big big data. Who take care? India still make data rule. If no strong rule, AI do bad.

India must all people together – govern, school, company and normal people. We teach AI and make sure AI not bad. Also talk to other country and learn from them.

If India use AI good way, we become strong, help poor and make better life. But if only rich use AI, and poor no get, then big bad thing happen.

So, in short, AI time in India have many hope and many danger. We must go right road. AI must help all people, not only some. Then India grow big and world say "good job India"."""

# Execute with an initial essay
initial_state = {
    'essay': essay2
}

# Run the workflow and print results
final_state = workflow.invoke(initial_state)
print("Essay Analysis Results:")
print(f"Average Score: {final_state['avg_score']:.2f}/10\n")
print("Language Feedback:")
print(final_state['language_feedback'])
print("\nAnalysis Feedback:")
print(final_state['analysis_feedback'])
print("\nClarity Feedback:")
print(final_state['clarity_feedback'])
print("\nOverall Evaluation:")
print(final_state['overall_feedback'])
