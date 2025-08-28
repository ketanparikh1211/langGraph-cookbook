"""
Twitter/X Post Generator Workflow using LangGraph

This script demonstrates iterative refinement in LangGraph by building a tweet generator
with evaluation and optimization in a feedback loop that continues until a tweet
meets quality standards or reaches a maximum number of iterations.

Key concepts:
- Iterative workflows with feedback loops
- Using structured output for evaluation
- Maintaining iteration history
- Conditional termination based on quality or iteration count
"""

from langgraph.graph import StateGraph,START, END
from typing import TypedDict, Literal, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import operator
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

# Initialize language models for different roles
generator_llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=api_key)
evaluator_llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=api_key)
optimizer_llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=api_key)

# Define schema for tweet evaluation
class TweetEvaluation(BaseModel):
    """Schema for tweet evaluation output.
    
    Attributes:
        evaluation: Final assessment ("approved" or "needs_improvement")
        feedback: Detailed feedback explaining the evaluation
    """
    evaluation: Literal["approved", "needs_improvement"] = Field(..., 
                                                               description="Final evaluation result.")
    feedback: str = Field(..., 
                         description="feedback for the tweet.")

# Create a structured output model for evaluation
structured_evaluator_llm = evaluator_llm.with_structured_output(TweetEvaluation)

# Define state structure for tweet generation workflow
class TweetState(TypedDict):
    """State definition for tweet generation workflow.
    
    Attributes:
        topic: The subject for the tweet
        tweet: Current version of the tweet
        evaluation: Evaluation result (approved/needs_improvement)
        feedback: Current evaluation feedback
        iteration: Current iteration count
        max_iteration: Maximum allowed iterations
        tweet_history: List of all generated tweets (aggregated with operator.add)
        feedback_history: List of all feedback (aggregated with operator.add)
    """
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int

    tweet_history: Annotated[list[str], operator.add]  # Special annotation for list aggregation
    feedback_history: Annotated[list[str], operator.add]  # Special annotation for list aggregation

def generate_tweet(state: TweetState):
    """Generate an initial tweet based on the given topic.
    
    Args:
        state: Current workflow state containing the topic
        
    Returns:
        Dictionary with generated tweet and added to history
    """
    # Create prompt for the tweet generator
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day english
""")
    ]

    # Generate the tweet
    response = generator_llm.invoke(messages).content

    # Return the tweet and add to history
    return {'tweet': response, 'tweet_history': [response]}

def evaluate_tweet(state: TweetState):
    """Evaluate the current tweet against quality criteria.
    
    Args:
        state: Current workflow state containing the tweet
        
    Returns:
        Dictionary with evaluation results and feedback added to history
    """
    # Create prompt for the evaluator
    messages = [
    SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
    HumanMessage(content=f"""
Evaluate the following tweet:

Tweet: "{state['tweet']}"

Use the criteria below to evaluate the tweet:

1. Originality – Is this fresh, or have you seen it a hundred times before?  
2. Humor – Did it genuinely make you smile, laugh, or chuckle?  
3. Punchiness – Is it short, sharp, and scroll-stopping?  
4. Virality Potential – Would people retweet or share it?  
5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 280 characters
- It reads like a traditional setup-punchline joke
- Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., "Masterpieces of the auntie-uncle universe" or vague summaries)

### Respond ONLY in structured format:
- evaluation: "approved" or "needs_improvement"  
- feedback: One paragraph explaining the strengths and weaknesses 
""")
]

    # Get structured evaluation response
    response = structured_evaluator_llm.invoke(messages)

    # Return evaluation results and add feedback to history
    return {
        'evaluation': response.evaluation, 
        'feedback': response.feedback, 
        'feedback_history': [response.feedback]
    }

def optimize_tweet(state: TweetState):
    """Improve the tweet based on evaluation feedback.
    
    Args:
        state: Current workflow state containing feedback
        
    Returns:
        Dictionary with improved tweet, incremented iteration count,
        and tweet added to history
    """
    # Create prompt for the optimizer
    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"

Topic: "{state['topic']}"
Original Tweet:
{state['tweet']}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")
    ]

    # Generate improved tweet
    response = optimizer_llm.invoke(messages).content
    
    # Increment iteration counter
    iteration = state['iteration'] + 1

    # Return improved tweet, updated iteration count, and add to history
    return {'tweet': response, 'iteration': iteration, 'tweet_history': [response]}

def route_evaluation(state: TweetState):
    """Determine whether to approve the tweet or continue improving it.
    
    Terminates the loop if:
    1. The tweet is approved by the evaluator
    2. Maximum iterations have been reached
    
    Args:
        state: Current workflow state with evaluation and iteration count
        
    Returns:
        String literal indicating which branch to follow
    """
    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    else:
        return 'needs_improvement'

# Create the workflow graph
graph = StateGraph(TweetState)

# Add nodes for each step
graph.add_node('generate', generate_tweet)
graph.add_node('evaluate', evaluate_tweet)
graph.add_node('optimize', optimize_tweet)

# Define the initial flow
graph.add_edge(START, 'generate')
graph.add_edge('generate', 'evaluate')

# Define the conditional branching based on evaluation
graph.add_conditional_edges(
    'evaluate',
    route_evaluation,
    {
        'approved': END,
        'needs_improvement': 'optimize'
    }
)

# Complete the feedback loop
graph.add_edge('optimize', 'evaluate')

# Compile the workflow
workflow = graph.compile()

# Example execution with initial topic
initial_state = {
    "topic": "demon slaying",
    "iteration": 1,
    "max_iteration": 5
}

# Run the workflow and print results
result = workflow.invoke(initial_state)

# Display the final results
print("\nTweet Generation Process")
print(f"Topic: {result['topic']}")
print(f"Final Tweet: {result['tweet']}")
print(f"Evaluation: {result['evaluation']}")
print(f"Final Feedback: {result['feedback']}")
print(f"Iterations Needed: {result['iteration']}")

# Display the evolution of tweets
print("\nTweet History:")
for i, tweet in enumerate(result['tweet_history']):
    print(f"\nVersion {i+1}:")
    print(tweet)