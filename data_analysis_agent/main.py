# Using LangChain agents, local LLM via Ollama,
# pandas for data handling, basic profiling,
# LLM-driven summarization of the dataset

import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns

from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain_ollama import ChatOllama
# from langchain.agents.agent_types import AgentType
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import OutputFixingParser

# -------------------------------
# DATASET LOADER AGENT
# -------------------------------

@tool
def load_dataset(file_path:str) -> str:
    """
    Load a dataset from a given file path.
    """
    if not os.path.exists(file_path):
        return "File not found."
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            return "Unsupported file format."

        # Save to global variable for other agents
        global dataset
        dataset = df

        return f"Dataset loaded with shape {df.shape} and columns: {','.join(df.columns)}"
    except Exception as e:
        return f"Error loading dataset: {str(e)}"


# -----------------------------------------
# DATA PROFILING AGENT
# -----------------------------------------
@tool
def profile_dataset(dummy_input: str) -> str:
    """Return summary statistics, null counts, and data types."""
    try:
        # Access global dataset
        global dataset
        profile = {
            "dtypes": dataset.dtypes.to_dict(),
            "null_counts": dataset.isnull().sum().to_dict(),
            "summary": dataset.describe(include="all").to_dict()
        }
        return str(profile)
    except Exception as e:
        return f"Error profiling dataset: {str(e)}"

# -----------------------------------------------------
# VISUALIZATION AGENT
# -----------------------------------------------------
@tool
def visualize_dataset(dummy_input: str) -> str:
    """Creates and saves basic visualizations: histograms and correlations."""
    try:
        global dataset
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)

        # Histogram for numeric columns
        num_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            plt.figure(figsize=(10, 4))
            dataset[col].hist(bins=20)
            plt.title(f"Histogram of {col}")
            plt.savefig(os.path.join(output_dir, f"{col}_histogram.png"))
            plt.close()

        # Correlation heatmap
        if len(num_cols) > 1:
            corr = dataset[num_cols].corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
            plt.close()

        return f"Saved visualization in {output_dir}"
    except Exception as e:
        return f"Error visualizing dataset: {str(e)}"



# -----------------------------------------------------
# SETUP LANGCHAIN AGENT WITH OLLAMA (MISTRAL/GEMMA:2B)
# -----------------------------------------------------

def setup_agent():
    llm = ChatOllama(model="mistral", temperature=0)
    tools = [load_dataset, profile_dataset, visualize_dataset]
    # memory = ConversationBufferMemory(memory_key="chat_history")

    # Custom prompt prefix
    prefix = (
        "You are a data analyst agent. "
        "Do not output raw Python code unless explicitly instructed to. "
        "You can load datasets, summarize them, and visualize them using the tools provided."
        "Always start by loading the dataset. Then you can profile or visualize it as needed."
        "Use the tools ONLY when needed and always follow the Action/Action Input format."
    )

    # Custom formatting instructions
    format_instructions = (
        "Use the following format:\n\n"
        "Question: the input question you must answer\n"
        "Thought: what you should do next\n"
        "Action: the action to take, should be one of [{tool_names}]\n"
        "Action Input: the input to the action\n"
        "Observation: the result of the action\n"
        "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
        "Thought: I now know the final answer\n"
        "Final Answer: the final answer to the original input question"
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": prefix,
            "format_instructions": format_instructions
        })
        # memory=memory

    if hasattr(agent, "llm_chain") and hasattr(agent.llm_chain, "output_parser"):
        parser = OutputFixingParser.from_llm(llm=llm, parser=agent.llm_chain.output_parser)
        agent.llm_chain.output_parser = parser

    return agent

# ------------------------------------------
# MAIN FUNCTION
# ------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Data Analysis Agent")
    parser.add_argument("--file", type=str, required=True, help="Path to dataset file")
    args = parser.parse_args()

    print("....Initializing agent...")
    agent = setup_agent()
    print("I'm ready to help you analyze your dataset. What would you like to do?")
    # while True:
    #     user_input = input("Enter your command (or 'exit' to quit): ")
    #     if user_input.lower() == 'exit':
    #         break
    #     response = agent.run(user_input + " " + args.file)
    #     print(response)

    print("Loading and analyzing dataset...")
    # prompt = (
    #     f"You are a data analyst agent. "
    #     f"Do not output raw Python code unless explicitly instructed to. "
    #     f"Instead, call the appropriate tool using only 'Action' and 'Action Input'.\n\n"
    #     f"Now: Load the file at {args.file} and summarize its structure, types, nulls, and key stats."
    # )
    response = agent.invoke("Load 'data/file.csv' and summarize it. Only use one tool at a time and follow the format strictly.", return_intermediate_steps=True)
    print(f"LLM-driven summary: {response}")
    print(response["intermediate_steps"])

    viz = agent.invoke("Generate visualization")
    print(f"Visualization output: {viz}")

if __name__ == "__main__":
    dataset = None # global shared dataset
    main()
