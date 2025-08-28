# LangGraph Learning Repository

This repository contains a series of examples demonstrating how to use [LangGraph](https://github.com/langchain-ai/langgraph) - a library for building stateful, multi-actor applications with LLMs.

## Overview

LangGraph extends LangChain's capabilities by enabling the creation of complex, multi-step workflows that can maintain state between steps. This repository provides progressive examples from basic state management to sophisticated agent interactions.

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd langgraph
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Example Descriptions

### 01_bmi_workflow.py
**Topic: Basic State Management**  
A simple example that calculates BMI based on weight and height, demonstrating the fundamental concepts of StateGraph, nodes, and edges in LangGraph.

### 02_llm_workflow.py
**Topic: Integrating Language Models**  
Shows how to incorporate a language model (OpenAI) into a LangGraph workflow to answer questions.

### 03_prompt_chaining.py
**Topic: Sequential LLM Calls**  
Demonstrates prompt chaining by creating a blog post generation workflow with separate outline and content creation steps.

### 04_batmans_workflow.py
**Topic: Parallel Processing**  
Illustrates parallel node execution by calculating multiple cricket statistics simultaneously before generating a summary.

### 05_essay_analyser.py
**Topic: Structured Output & Aggregation**  
Shows how to use structured outputs with Pydantic models to analyze essays from multiple perspectives and aggregate the results.

### 06_quadratic_eqn.py
**Topic: Conditional Branching**  
Implements a quadratic equation solver with different paths based on the discriminant value, demonstrating conditional edges.

### 07_review_reply.py
**Topic: Advanced Branching Logic**  
Creates a customer review response system that analyzes sentiment and generates appropriate replies based on the analysis.

### 08_X_post_generator.py
**Topic: Iterative Refinement**  
Builds a tweet generator with evaluation and optimization in a feedback loop, showing how to implement iterative workflows.

### 09_chatbot.py
**Topic: Message Handling**  
Introduces message-based state management for creating a simple chatbot with memory.

### 10_persistence.py
**Topic: State Persistence**  
Demonstrates how to implement persistence in LangGraph to save and retrieve workflow state across multiple sessions.

### 11_tools.py
**Topic: Tool Integration**  
Shows how to incorporate external tools (search, calculator, stock price API) into a LangGraph workflow for enhanced capabilities.

## Key Concepts

- **StateGraph**: The core component for defining workflow structure
- **Nodes**: Functions that process and transform state
- **Edges**: Connections that define the flow between nodes
- **Conditional Routing**: Logic for determining the next node based on the current state
- **State Persistence**: Saving and retrieving workflow progress
- **Tool Integration**: Incorporating external capabilities into LLM workflows

## Resources

- [LangGraph Documentation](https://github.com/langchain-ai/langgraph)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI API Documentation](https://platform.openai.com/docs/introduction)