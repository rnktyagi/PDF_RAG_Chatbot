from typing import TypedDict , Annotated , List
from langgraph.graph.message import add_messages 
from langchain_core.messages import SystemMessage , HumanMessage
from langchain_openai import ChatOpenAI
import os
from src.retrieval import Retriever
import os
from tavily import TavilyClient
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START ,END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

load_dotenv()

class State(TypedDict) :
    messages : Annotated[list , add_messages]
    context : List[dict]
    rewritten_query : str
    user_id : str
    web_search_needed : bool
    retry : int 

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)

retriever = Retriever()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def rewrite_node(state : State) :
    messages = state['messages']

    # 1. Filter to only get the human's messages
    user_msg = [msg for msg in messages if isinstance(msg , HumanMessage)]

    # 2. Extract the actual text
    latest_ques = user_msg[-1].content
    history = "\n".join([msg.content for msg in user_msg[:-1]])

    # 3. Set the strict system rules
    system_prompt = SystemMessage(content="""You are an expert search query generator for a vector database. 
Your ONLY job is to convert the user's latest input into a single, highly optimized search string.

You will receive a sequence of the user's previous questions, followed by their newest input. 

CRITICAL RULES:
1. TRACK THE TRAIN OF THOUGHT: If the latest input uses pronouns (it, they, this) or is a fragment (e.g., "What about the budget?"), identify the core noun from the previous questions and substitute it.
2. NO CONVERSATIONAL FILLER: Do not answer the question. Do not explain your reasoning.
3. FORMAT: Output only the raw search keywords. No commas, no bullet points.

Example Input:
Chat History: 
What is the main objective of Project Chronos?
Who is the lead engineer?
Latest User Input: What is his total budget for Q4?

Example Output: Project Chronos lead engineer budget
""")

    # 4. FIX: Package the history and question into a proper HumanMessage object
    human_prompt = HumanMessage(content=f"Chat History: {history}\n\nLatest User Input: {latest_ques}\n\nGenerate the concise search query now:")

    # 5. FIX: Combine them as a valid list of Message objects
    final_msg = [system_prompt, human_prompt]

    # 6. Invoke the LLM
    response = llm.invoke(final_msg)

    print("\n" + "="*60, flush=True)
    print(f"\n ReQuery : \n{response.content} \n", flush=True)
    print("="*60 + "\n", flush=True)

    return {'rewritten_query' : response.content}

def retrieve_node(state : State) :
    user_id = state['user_id']
    re_query = state['rewritten_query']

    context = retriever.retrieve(re_query , user_id)

    return{'context' : context}

def answer_node(state : State) :
    messages = state['messages']
    context = state['context']
    retry = state.get('retry' , 0)

    context_text = ""
    if not context:
        context_text = "No relevant context found in the database for this specific query."
    else:
        for i, chunk in enumerate(context):
            context_text += f"\n--- Document Chunk {i+1} ---\n"
            context_text += f"Source: {chunk.get('source', 'Unknown')}\n"
            context_text += f"Pages: {chunk.get('pages', 'N/A')}\n"
            context_text += f"Section: {chunk.get('section', 'N/A')}\n"
            context_text += f"Content: {chunk.get('text', '')}\n"

    print("\n" + "="*60, flush=True)
    print(f"\n\nCONTEXT TEXT :/n/n{context_text}", flush=True)
    print("="*60 + "\n", flush=True)

    if retry<1 :
        system_prompt = SystemMessage(content=f"""
        You are an advanced enterprise RAG assistant. Your job is to answer the user's latest question 
        by strictly analyzing the conversation history and the provided document chunks below.
        
        CRITICAL RULES:
        1. Base your answer ONLY on the text snippets provided in the Context section below. Do not assume or extrapolate.
        2. If the context does not contain the answer, or if the context is irrelevant to the question, 
        you must reply with exactly this phrase and absolutely nothing else: FALLBACK_TO_WEB_SEARCH
        3. You MUST inline cite your sources whenever you use information from a chunk. 
        Format your citations cleanly at the end of sentences like this: [Source: file.pdf, Page: X].
        
        CONTEXT DATA:
        {context_text}
        """)
    else :
        system_prompt = f"""
        You are an advanced enterprise RAG assistant. Your job is to answer the user's latest question 
        by strictly analyzing the conversation history and the provided document chunks below. 
        These chunks now include both internal documents and live web search results.
        
        CRITICAL RULES:
        1. Base your answer ONLY on the text snippets provided in the Context section below. Do not assume or extrapolate.
        2. DO NOT ask for another web search. If the answer is still not found in the provided context, you must politely inform the user that the information is unavailable.
        3. You MUST inline cite your sources whenever you use information from a chunk. 
           Format your citations cleanly at the end of sentences like this: [Source: file.pdf, Page: X] or [Source: website_url].
        
        CONTEXT DATA:
        {context_text}
        """

    final_msg = [system_prompt] + messages

    response = llm.invoke(final_msg)

    if response.content.strip() == "FALLBACK_TO_WEB_SEARCH":
        return {"web_search_needed": True}
    else:
        return {"messages": [response], 
                "web_search_needed": False}

def routing(state : State) :
    if state["web_search_needed"] :
        return "web_search_node"
    else:
        return "END"
    
def web_search_node(state : State) :
    re_query = state['rewritten_query']
    context = state['context']
    retry = state.get('retry' , 0)

    response = tavily_client.search(query=re_query , max_results=3)
    results = response['results']

    web_context = []

    for res in results :
        web_context.append({
            "text": res.get("content", ""),
            "source": res.get("url", "Live Web Search"),
            "pages": "N/A",
            "section": "Internet Result"
        })

    combined = context + web_context

    return {'context' : combined , 'retry' : retry+1}

workflow = StateGraph(State)

workflow.add_node("rewrite_node" , rewrite_node)
workflow.add_node("retrieve_node" , retrieve_node)
workflow.add_node("answer_node" , answer_node)
workflow.add_node("web_search_node" , web_search_node)

workflow.add_edge(START , "rewrite_node")
workflow.add_edge("rewrite_node" , "retrieve_node")
workflow.add_edge("retrieve_node" , "answer_node")
workflow.add_conditional_edges(
    "answer_node",
    routing,
    {"web_search_node": "web_search_node",
    "END": END})
workflow.add_edge("web_search_node" , "answer_node")

