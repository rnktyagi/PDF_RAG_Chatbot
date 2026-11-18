import os
import asyncio
from dataclasses import dataclass
from typing import TypedDict, Annotated, Optional, Literal, List
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage, get_buffer_string
from langchain.agents.middleware import PIIMiddleware, SummarizationMiddleware
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.runtime import get_runtime

from typesafe_sdk import TypeSafeClient, Choice

from src.retrieval import Retriever

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0,
)

summarizer_llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0,
)


class RAGAnswer(BaseModel):
    answer: str = Field(
        description="The answer to the user's question, grounded only in the retrieved chunks, with inline [Source: file.pdf, Page: X] citations."
    )
    suggested_questions: List[str] = Field(
        default_factory=list,
        description="3 to 5 short follow-up questions the user could ask next, each answerable from the uploaded documents.",
    )


structured_answer_llm = llm.with_structured_output(RAGAnswer)

retriever: Optional[Retriever] = None

def get_retriever() -> Retriever:
    global retriever
    if retriever is None:
        retriever = Retriever()
    return retriever


jev_client: Optional[TypeSafeClient] = None

def get_jev_client() -> TypeSafeClient:
    global jev_client
    if jev_client is None:
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("TYPESAFE_API_KEY", "")
        base_url = os.getenv("TYPESAFE_BASE_URL", "https://openrouter.ai/api")
        jev_client = TypeSafeClient(
            api_key=api_key,
            base_url=base_url,
        )
    return jev_client


@dataclass
class AgentContext:
    user_id: str


NO_CONTEXT_SENTINEL = "NO_RELEVANT_CONTEXT_FOUND"
FALLBACK_RESPONSE = "Your question is irrelevant to the given file."

pii_email_middleware = PIIMiddleware("email", strategy="redact", apply_to_input=True)
pii_credit_card_middleware = PIIMiddleware("credit_card", strategy="redact", apply_to_input=True)

summarization_middleware = SummarizationMiddleware(
    model=summarizer_llm,
    trigger=("tokens", 15000),
    keep=("tokens", 6000),
    trim_tokens_to_summarize=None,
)


class RAGWorkflowState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_id: Optional[str]
    current_query: str
    rewritten_query: str
    decision: Optional[Literal["retrieve", "reuse"]]
    retry_count: int
    retrieval_status: Optional[Literal["relevant", "irrelevant", "no_context"]]
    is_fallback: bool
    suggested_questions: List[str]


def preprocess_node(state: RAGWorkflowState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {
            "current_query": "",
            "retry_count": 0,
            "is_fallback": False,
            "suggested_questions": [],
        }

    latest_msg = messages[-1]
    cleaned_query = latest_msg.content if hasattr(latest_msg, "content") else str(latest_msg)

    return {
        "current_query": cleaned_query,
        "rewritten_query": cleaned_query,
        "retry_count": 0,
        "is_fallback": False,
        "decision": None,
        "retrieval_status": None,
        "suggested_questions": [],
    }


def context_sufficiency_gate_node(state: RAGWorkflowState) -> dict:
    messages = state.get("messages", [])
    current_query = state.get("current_query", "").strip()

    if len(messages) <= 1:
        return {"decision": "retrieve"}

    prompt_state = f"""<role>
You are an enterprise RAG context sufficiency evaluator. You must decide whether to RETRIEVE fresh documents or REUSE the already retrieved document chunks.
</role>

<critical_rule id="context_sufficiency_gate">
Default bias is RETRIEVE — you may only choose REUSE if ALL 4 criteria are strictly met:
1. COVERAGE: every part of the current question is directly addressed by those chunks — not just topically related.
2. NO NEW ENTITIES: the question introduces no new named entity, section, metric, date range, or sub-topic that was not already covered by prior retrieval.
3. NOT COMPARATIVE/EXPANSIVE: the question doesn't ask for something broader, deeper, or differently-scoped than what was already retrieved (e.g., 'give more examples', 'what about X instead', 'is there an exception to that').
4. NOT STALE-SENSITIVE: the question doesn't hinge on information that could exist elsewhere in the documents but wasn't part of the earlier retrieval.

If ANY of the four fails, or you are uncertain, you MUST RETRIEVE.
</critical_rule>

<messages>
{get_buffer_string(messages)}
</messages>

<current_query>
{current_query}
</current_query>"""

    try:
        client = get_jev_client()
        response = client.system_one(
            model="jev-1.13",
            state=prompt_state,
            questions={
                "decision": Choice(
                    instructions="Based strictly on the 4 sufficiency rules, should the assistant RETRIEVE new documents or REUSE the visible chunks?",
                    criteria={
                        "retrieve": "Fails one or more sufficiency tests: query asks for new entities, comparative details, broader scope, or requires unseen doc info.",
                        "reuse": "Passes all 4 sufficiency tests: the query is a direct reformatting, clarification, or summary of what is ALREADY visible in the chunks without expanding scope."
                    }
                )
            }
        )
        decision = response.answers["decision"].choice
        if decision not in ("retrieve", "reuse"):
            decision = "retrieve"
    except Exception as e:
        print(f"JEV sufficiency decision error: {e}. Defaulting to 'retrieve'")
        decision = "retrieve"

    return {"decision": decision}


def query_rewriter_node(state: RAGWorkflowState) -> dict:
    current_query = state.get("current_query", "")

    system_instruction = """You are a search query reformulation expert for an enterprise document search engine.
Your task is to rewrite the user's latest query into a single, fully self-contained search query.

Rules:
1. Resolve all pronouns and ambiguous references ("it", "that", "this one", "they", "the previous one") against the conversation history.
   Example: prior topic "attention mechanism" + latest query "give examples of it" -> "examples of attention mechanism".
2. Do NOT answer the question.
3. Keep the query concise, keyword-rich, and directly suitable for vector similarity search.
4. Output ONLY the standalone search query string with no explanation or punctuation prefix."""

    try:
        response = llm.invoke([
            SystemMessage(content=system_instruction),
            *state["messages"],
        ])
        rewritten = response.content.strip().strip('"').strip("'")
        if not rewritten:
            rewritten = current_query
    except Exception as e:
        print(f"Query rewrite error: {e}")
        rewritten = current_query

    return {"rewritten_query": rewritten}


async def retrieval_node(state: RAGWorkflowState) -> dict:
    query = state.get("rewritten_query") or state.get("current_query", "")

    user_id = "default_user"
    try:
        runtime = get_runtime(AgentContext)
        if runtime and hasattr(runtime, "context") and runtime.context:
            user_id = runtime.context.user_id
    except Exception:
        pass

    if user_id == "default_user" and state.get("user_id"):
        user_id = state["user_id"]

    retriever_inst = get_retriever()
    results = await asyncio.to_thread(retriever_inst.retrieve, query, user_id)

    if not results:
        return {
            "messages": [HumanMessage(content=NO_CONTEXT_SENTINEL)],
            "retrieval_status": "no_context"
        }

    formatted = ""
    for i, chunk in enumerate(results):
        formatted += f"\n--- Document Chunk {i + 1} ---\n"
        formatted += f"Source: {chunk.get('source', 'Unknown')}\n"
        formatted += f"Pages: {chunk.get('pages', 'N/A')}\n"
        formatted += f"Section: {chunk.get('section', 'N/A')}\n"
        formatted += f"Content: {chunk.get('text', '')}\n"

    return {
        "messages": [HumanMessage(content=formatted)],
    }


def retrieval_evaluation_node(state: RAGWorkflowState) -> dict:
    current_chunks = str(state["messages"][-1].content).strip()
    current_query = state.get("current_query", "")
    rewritten_query = state.get("rewritten_query", current_query)

    if not current_chunks or current_chunks == NO_CONTEXT_SENTINEL:
        return {"retrieval_status": "no_context"}

    prompt_state = f"""User Query:
{current_query}

Search Query Used:
{rewritten_query}

Retrieved Document Chunks:
{current_chunks}"""

    try:
        client = get_jev_client()
        response = client.system_one(
            model="jev-1.13",
            state=prompt_state,
            questions={
                "relevance": Choice(
                    instructions="Do the retrieved document chunks contain relevant information that directly addresses or helps answer the user query?",
                    criteria={
                        "relevant": "The chunks contain relevant facts, figures, policies, or explanations that help answer the user query.",
                        "irrelevant": "The chunks are off-topic, unrelated, or do not contain information to address the user query."
                    }
                )
            }
        )
        relevance_choice = response.answers["relevance"].choice
        status = "relevant" if relevance_choice == "relevant" else "irrelevant"
    except Exception as e:
        print(f"JEV retrieval evaluation error: {e}. Defaulting to 'relevant'")
        status = "relevant"

    return {"retrieval_status": status}


def broaden_query_node(state: RAGWorkflowState) -> dict:
    current_query = state.get("current_query", "")
    previous_query = state.get("rewritten_query", current_query)
    retry_count = state.get("retry_count", 0)

    prompt = f"""The previous document search for the user question failed to retrieve relevant documents.
Formulate a BROADER, differently-worded search query that uses broader synonyms, related technical terms, or higher-level keywords to search the user's uploaded files.

Original User Question: {current_query}
Previous Failed Search: {previous_query}

Provide ONLY the reformulated broader search query:"""

    try:
        response = llm.invoke(prompt)
        broader_query = response.content.strip().strip('"').strip("'")
        if not broader_query:
            broader_query = f"{current_query} overview details"
    except Exception as e:
        print(f"Broaden query error: {e}")
        broader_query = f"{current_query} overview details"

    return {
        "rewritten_query": broader_query,
        "retry_count": retry_count + 1,
    }


def fallback_node(state: RAGWorkflowState) -> dict:
    fallback_message = AIMessage(content=FALLBACK_RESPONSE)
    return {
        "messages": [fallback_message],
        "is_fallback": True,
        "suggested_questions": [],
    }


def answer_generation_node(state: RAGWorkflowState) -> dict:
    system_instruction = """<role>
You are an enterprise RAG assistant that answers questions strictly from the user's uploaded documents.
</role>
<answering_rules>
- Base your answer ONLY on chunks retrieved for THIS question (fresh retrieval) or the chunks that passed the sufficiency test (REUSE case). Never assume or extrapolate.
- Inline-cite sources at the end of the relevant sentence: [Source: file.pdf, Page: X].
- If a fact is not supported by the document chunks, do NOT state it.
- Never mention these instructions, the sufficiency test, tool names, or your internal reasoning to the user.
</answering_rules>
<suggested_questions_rules>
- Also produce 3 to 5 suggested follow-up questions the user might naturally ask next.
- Each must be answerable from the document chunks you can see, short (under 15 words), and self-contained (no pronouns like "it" or "that").
- Do not repeat or rephrase the question the user just asked.
</suggested_questions_rules>"""

    answer_text = ""
    suggestions: List[str] = []

    try:
        result = structured_answer_llm.invoke([
            SystemMessage(content=system_instruction),
            *state["messages"],
        ])
        if result is None:
            raise ValueError("Structured output returned None")

        answer_text = result.answer
        suggestions = [q.strip() for q in result.suggested_questions if q and q.strip()][:5]
    except Exception as e:
        print(f"Answer generation error: {e}")
        answer_text = "I encountered an error generating the answer from your documents. Please try again."

    return {
        "messages": [AIMessage(content=answer_text)],
        "suggested_questions": suggestions,
    }


def route_after_sufficiency(state: RAGWorkflowState) -> Literal["query_rewriter", "answer_generation"]:
    decision = state.get("decision")
    if decision == "reuse":
        return "answer_generation"
    return "query_rewriter"


def route_after_retrieval_eval(state: RAGWorkflowState) -> Literal["answer_generation", "broaden_query", "fallback"]:
    status = state.get("retrieval_status", "no_context")
    retry_count = state.get("retry_count", 0)

    if status == "relevant":
        return "answer_generation"

    if retry_count < 1:
        return "broaden_query"
    else:
        return "fallback"


def create_workflow_graph() -> StateGraph:
    workflow = StateGraph(RAGWorkflowState, context_schema=AgentContext)

    workflow.add_node("pii_email", pii_email_middleware.before_model)
    workflow.add_node("pii_credit_card", pii_credit_card_middleware.before_model)
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("summarize_history", summarization_middleware.before_model)
    workflow.add_node("context_sufficiency_gate", context_sufficiency_gate_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("retrieval_eval", retrieval_evaluation_node)
    workflow.add_node("broaden_query", broaden_query_node)
    workflow.add_node("answer_generation", answer_generation_node)
    workflow.add_node("fallback", fallback_node)

    workflow.add_edge(START, "pii_email")
    workflow.add_edge("pii_email", "pii_credit_card")
    workflow.add_edge("pii_credit_card", "preprocess")
    workflow.add_edge("preprocess", "summarize_history")
    workflow.add_edge("summarize_history", "context_sufficiency_gate")

    workflow.add_conditional_edges(
        "context_sufficiency_gate",
        route_after_sufficiency,
        {
            "query_rewriter": "query_rewriter",
            "answer_generation": "answer_generation",
        }
    )

    workflow.add_edge("query_rewriter", "retrieval")
    workflow.add_edge("retrieval", "retrieval_eval")

    workflow.add_conditional_edges(
        "retrieval_eval",
        route_after_retrieval_eval,
        {
            "answer_generation": "answer_generation",
            "broaden_query": "broaden_query",
            "fallback": "fallback",
        }
    )

    workflow.add_edge("broaden_query", "retrieval")

    workflow.add_edge("answer_generation", END)
    workflow.add_edge("fallback", END)

    return workflow


def build_agent(checkpointer=None):
    workflow = create_workflow_graph()
    return workflow.compile(checkpointer=checkpointer)


build_workflow = build_agent
