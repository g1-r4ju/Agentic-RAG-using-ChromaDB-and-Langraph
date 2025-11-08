from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence, Literal, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
import os

persist_dir = "./chroma_langchain_db"
collection_name = "rag-chroma"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load or create vectorstore
if not os.path.exists(persist_dir):
    print("Vectorstore not found — creating and embedding documents...")
    documents = DirectoryLoader("data", glob="**/*.pdf", show_progress=True).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(
        documents=texts,
        collection_name=collection_name,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
else:
    print("Loading existing vectorstore...")
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embedding_model,
    )

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3} 
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool retrieves relevant information from the RAG system based on the user's query.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)

tools = [retriever_tool]

# Initialize LLM with tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
llm_with_tools = llm.bind_tools(tools)

# Define graph state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    relevancy: bool
    
# Define the assistant's behavior and usage instructions
system_prompt = """
You are an intelligent AI assistant who answers questions based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

relevancy_prompt = '''
Verify whether the question is relevant to our documents. List of documents we have are of Monopoly, Chess and UNO game rules.
If they are relevant, answer with "yes", otherwise answer with "no".
Here is the question: 
{question}
'''

def check_relevancy(state: AgentState) -> AgentState:
    """Check if the question is relevant to the documents."""
    relevancy_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Get the last user message
    last_message = state['messages'][-1]
    question = last_message.content
    
    # Format the relevancy check prompt
    prompt = relevancy_prompt.format(question=question)
    response = relevancy_llm.invoke(prompt)
    
    # Update state with relevancy
    is_relevant = 'yes' in response.content.lower()
    return {'relevancy': is_relevant}

def node_call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    
    # Add system prompt at the beginning if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + messages
    
    # Call LLM with tools
    message = llm_with_tools.invoke(messages)
    return {'messages': [message]}

def node_take_action(state: AgentState) -> AgentState:
    """Execute tool calls requested by the LLM."""
    last_message = state['messages'][-1]
    
    # Check if the last message has tool calls
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return state
    
    tool_calls = last_message.tool_calls
    tools_dict = {tool.name: tool for tool in tools}
    
    results = []
    for t in tool_calls:
        # Get the tool and invoke it
        tool_name = t['name']
        tool_args = t['args']
        
        if tool_name in tools_dict:
            result = tools_dict[tool_name].invoke(tool_args)
            # Wrap the tool response as a ToolMessage
            results.append(
                ToolMessage(
                    tool_call_id=t['id'],
                    name=tool_name,
                    content=str(result)
                )
            )
    
    return {'messages': results}

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Check if the most recent LLM message contains tool calls."""
    last_message = state['messages'][-1]
    
    # If it's an AI message with tool calls, continue to tools
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end the conversation
    return "end"

def route_relevancy(state: AgentState) -> Literal["continue", "not_relevant"]:
    """Route based on relevancy check."""
    if state.get('relevancy', False):
        return "continue"
    else:
        return "not_relevant"

def handle_not_relevant(state: AgentState) -> AgentState:
    """Handle cases where the question is not relevant to the documents."""
    not_relevant_message = AIMessage(
        content="I'm sorry, but your question doesn't appear to be related to the documents I have access to. "
                "I can help you with questions about Monopoly, Chess, and UNO game rules. "
                "Please ask a question related to these topics."
    )
    return {'messages': [not_relevant_message]}

# Build the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("check_relevancy", check_relevancy)
workflow.add_node("llm", node_call_llm)
workflow.add_node("tools", node_take_action)
workflow.add_node("not_relevant", handle_not_relevant)

# Add edges
workflow.add_edge(START, "check_relevancy")
workflow.add_conditional_edges(
    "check_relevancy",
    route_relevancy,
    {
        "continue": "llm",
        "not_relevant": "not_relevant"
    }
)
workflow.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "llm")
workflow.add_edge("not_relevant", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    user_input = input('Enter the question: ')
    messages = [HumanMessage(content=user_input)]
    print("USER:\n", user_input)
    print("-----")

    thread_id = 1
    config = {"configurable": {"thread_id": thread_id}}
    
    result = app.invoke({"messages": messages}, config=config)
    
    final_message = result['messages'][-1]
    print("ANSWER:\n", final_message.content)
