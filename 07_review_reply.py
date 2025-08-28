"""
Customer Review Response Workflow using LangGraph

This script demonstrates advanced branching logic in LangGraph by creating
a customer review response system that analyzes sentiment and generates
appropriate replies based on the analysis.

Key concepts:
- Using structured output with Pydantic models
- Multi-path conditional branching
- Two-stage analysis with sentiment and detailed diagnosis
"""

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
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

# Define Pydantic schemas for structured outputs
class SentimentSchema(BaseModel):
    """Schema for sentiment analysis output.
    
    Attributes:
        sentiment: Classification as 'positive' or 'negative'
    """
    sentiment: Literal["positive", "negative"] = Field(description='Sentiment of the review')

class DiagnosisSchema(BaseModel):
    """Schema for detailed review diagnosis.
    
    Attributes:
        issue_type: Category of issue mentioned in the review
        tone: Emotional tone expressed by the user
        urgency: How critical the issue appears to be
    """
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(
        description='The category of issue mentioned in the review'
    )
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(
        description='The emotional tone expressed by the user'
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description='How urgent or critical the issue appears to be'
    )

# Create structured output models
structured_model = model.with_structured_output(SentimentSchema)
structured_model2 = model.with_structured_output(DiagnosisSchema)

# Define the state structure for our review response workflow
class ReviewState(TypedDict):
    """State definition for review response workflow.
    
    Attributes:
        review: The customer review text
        sentiment: Classified sentiment (positive/negative)
        diagnosis: Detailed analysis of negative reviews
        response: Generated response to the review
    """
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str

def find_sentiment(state: ReviewState):
    """Analyze the sentiment of a customer review.
    
    Args:
        state: Current workflow state containing the review
        
    Returns:
        Dictionary with sentiment classification
    """
    prompt = f'For the following review find out the sentiment \n {state["review"]}'
    sentiment = structured_model.invoke(prompt).sentiment

    return {'sentiment': sentiment}

def check_sentiment(state: ReviewState) -> Literal["positive_response", "run_diagnosis"]:
    """Determine which branch to take based on sentiment.
    
    Args:
        state: Current workflow state with sentiment classification
        
    Returns:
        String literal indicating which branch to follow
    """
    if state['sentiment'] == 'positive':
        return 'positive_response'
    else:
        return 'run_diagnosis'
    
def positive_response(state: ReviewState):
    """Generate a response to a positive review.
    
    Args:
        state: Current workflow state with positive sentiment
        
    Returns:
        Dictionary with generated response
    """
    prompt = f"""Write a warm thank-you message in response to this review:
    \n\n\"{state['review']}\"\n
Also, kindly ask the user to leave feedback on our website."""
    
    response = model.invoke(prompt).content

    return {'response': response}

def run_diagnosis(state: ReviewState):
    """Analyze a negative review in detail.
    
    Extracts issue type, emotional tone, and urgency to guide
    the response generation.
    
    Args:
        state: Current workflow state with negative sentiment
        
    Returns:
        Dictionary with detailed diagnosis
    """
    prompt = f"""Diagnose this negative review:\n\n{state['review']}\n"
    "Return issue_type, tone, and urgency.
"""
    response = structured_model2.invoke(prompt)

    return {'diagnosis': response.model_dump()}

def negative_response(state: ReviewState):
    """Generate a tailored response to a negative review.
    
    Uses the detailed diagnosis to create an appropriate,
    empathetic response addressing the specific issues.
    
    Args:
        state: Current workflow state with diagnosis results
        
    Returns:
        Dictionary with generated response
    """
    diagnosis = state['diagnosis']

    prompt = f"""You are a support assistant.
The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
Write an empathetic, helpful resolution message.
"""
    response = model.invoke(prompt).content

    return {'response': response}

# Create the workflow graph
graph = StateGraph(ReviewState)

# Add nodes for each step in the workflow
graph.add_node('find_sentiment', find_sentiment)
graph.add_node('positive_response', positive_response)
graph.add_node('run_diagnosis', run_diagnosis)
graph.add_node('negative_response', negative_response)

# Define the initial flow
graph.add_edge(START, 'find_sentiment')

# Define conditional branching based on sentiment
graph.add_conditional_edges(
    'find_sentiment',
    check_sentiment,
    {
        'positive_response': 'positive_response',
        'run_diagnosis': 'run_diagnosis'
    }
)

# Define the remaining flows
graph.add_edge('positive_response', END)
graph.add_edge('run_diagnosis', 'negative_response')
graph.add_edge('negative_response', END)

# Compile the workflow
workflow = graph.compile()

# Example negative review
initial_state = {
    'review': "I've been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality."
}

# Run the workflow and print results
response = workflow.invoke(initial_state)
print("Customer Review:")
print(response['review'])
print("\nSentiment Analysis:")
print(response['sentiment'])

if response['sentiment'] == 'negative':
    print("\nDetailed Diagnosis:")
    print(f"Issue Type: {response['diagnosis']['issue_type']}")
    print(f"Tone: {response['diagnosis']['tone']}")
    print(f"Urgency: {response['diagnosis']['urgency']}")

print("\nGenerated Response:")
print(response['response'])