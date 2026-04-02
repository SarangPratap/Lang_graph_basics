
import hashlib
import sys
from typing import List, Optional
import os
from langchain.messages import HumanMessage
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
from pydantic import BaseModel
from chains import generation_chain, reflection_chain, summarize_chain, research_chain, human_input_chain
from pypdf import PdfReader
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from llama_index.core import Document

embedding_model = OllamaEmbedding(
    model_name="llama3",
)


load_dotenv() 

REFLECT = "reflect"
GENERATE = "generate"
RESEARCH = "research"
HUMAN_REVISION = "human_revision"
SUMMARIZE = "summarize"



class Graph_state(BaseModel):
    job_description: str = ""
    research_brief: Optional[str] = None
    summarize: Optional[str] = None
    current_draft : Optional[str] = None
    critique: Optional[str] = None
    human_feedback: Optional[str] = None
    revision_brief: Optional[str] = None
    retrieved_context: Optional[List[str]] = None
    iterations: int = 0
    
    


def research_node(graph_state: Graph_state) -> Graph_state:
    # we will call the research prompt here and update the graph state with the research brief
    resume_context = "\n\n".join(graph_state.retrieved_context or [])
    research_input = HumanMessage(
        content=f"Job Description:\n{graph_state.job_description}\n\nResume:\n{resume_context}"
    )
    
    research_brief = research_chain.invoke({
        "messages": [research_input]
    })
    
    return {
        
        "research_brief": research_brief,
        "retrieved_context": graph_state.retrieved_context    
            
    }


def summarize_node(state: Graph_state):
    
    summarize_input = HumanMessage(
        content=(
             "Research Brief:\n"
             f"{state.research_brief}\n\n"
             "Current Draft:\n"
             f"{state.retrieved_context}\n\n"
        ))
    
    
    summary = summarize_chain.invoke({
        "messages": [summarize_input]
    })
    
    return {
        
            "summarize": summary,
    }


#The Logic: A Node is just a Python function that takes the current state, 
# does some work (calls Llama 3), and returns the update.
def generate_node(state: Graph_state) -> dict: # every node receives the entire state
    parts = []
    if state.summarize:
        parts.append(f"Summarize: {state.summarize}")
    if state.critique:
        parts.append(f"Critique: {state.critique}")
    if state.revision_brief:
        parts.append(f"Human Revision Brief: {state.revision_brief}")
    if not parts:
        parts.append(f"Create the First Draft: [from this resume , {state.current_draft}, based on the Job Description: {state.job_description}]")
    
    generate_input = HumanMessage(
        content=("\n\n".join(parts)))
    
    rew_resume = generation_chain.invoke({ 
                                          
                                          "messages": [generate_input]
                                      
                                        })  # the messages varible comes from MessagesPlaceholder(variable_name="messages")
    # LangGraph will 'add' this one message to your 'chat_history'
    return {
        "current_draft": rew_resume,
        "iterations": state.iterations + 1,
        }  # we return the entire chat history with the new message appended at the end. This is important for reflection to work properly, as it needs the full context of the conversation to provide meaningful feedback.


def human_revision_node(state: Graph_state) -> dict:
    print("\n=== CURRENT DRAFT ===\n")
    print(state.current_draft or "(No draft available)")

    print("\n=== REFLECTION CRITIQUE ===\n")
    print(state.critique or "(No critique available)")

    human_feedback = input("Enter your feedback for the resume draft (or press Enter to skip): ")
    state.human_feedback = human_feedback.strip() if human_feedback else None
    
    if not state.human_feedback:  # if there is no feedback from human
        return {"revision_brief": state.revision_brief}

    revision_brief = human_input_chain.invoke({
        "critique": state.critique or "",
        "human_feedback": state.human_feedback,
    })

    return {
        "revision_brief": revision_brief,
    }

def reflect_node(state: Graph_state) -> dict:
    # 1. Pass the FULL chat_history so the Recruiter sees the JD AND the Draft
    
    reflect_input = HumanMessage(
        content=(
            "Job Description:\n"
            f"{state.job_description}\n\n"
            "Current Draft:\n"
            f"{state.current_draft}\n\n"
        )
    )
    
    critique = reflection_chain.invoke({
        
        "messages": [reflect_input]  #Dot notataion for pydantic
    })
    return {
        
        "critique": critique
            
            }



def resume_file_hash(file_path: str) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_resume_documents(file_path: str):
    # we will load the resume file and convert it into documents that can be ingested by the index
    # for simplicity, we will just read the file and return it as a single document
    
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text())
    text = "\n\n".join(text_parts)
    
    return [Document(text=text, metadata={"source": os.path.basename(file_path)})]


#Ingestion pipelien
def ingestion_pipeline(path: str, persist_dir= "./llamaidx_db", model=embedding_model) -> VectorStoreIndex:
    # Step 1: Create the index
    
    h_val= resume_file_hash(path)
    persist_dir= os.path.join(persist_dir, h_val)
    
    documents = load_resume_documents(path)
    
    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        print("Index folder exists and is not empty -> load index")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        
        return load_index_from_storage(storage_context, embed_model = model)
    else:
        print("No usable index -> build new index")
        index = VectorStoreIndex.from_documents(documents,  embed_model = model)
        index.storage_context.persist(persist_dir=persist_dir)
        return index



def query_pipeline(index: VectorStoreIndex, job_description: str, top_k: int) -> str:
    """Retrieve top resume chunks matching the job description."""
    retriever = index.as_retriever(similarity_top_k=top_k)
    print(retriever)
    nodes = retriever.retrieve(job_description)
    print(f"Retrieved {len(nodes)} context chunks from resume.")
    return [node.node.get_content() for node in nodes]

def conditional_revision(state: Graph_state) -> str:
    MAX_ITERATIONS = 3
    # This function determines whether to loop back to generation based on the presence of human feedback
    if state.iterations >= MAX_ITERATIONS:  # Max iterations to prevent infinite loops
        return END
    else:
        return "continue"  # Add edge for human revision



if __name__ =="__main__":
    # Example usage

    
    
    graph = StateGraph(Graph_state)
    resume_path = sys.argv[1]
    print("Enter job description (paste text and press Enter): ")
    try:
        job_description = input().strip()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)

    # Save job description to file to avoid PowerShell issues
    with open("job_description.txt", "w", encoding="utf-8") as f:
        f.write(job_description)
    
    print("Job description received. Processing...\n")
    
    index = ingestion_pipeline(resume_path)
    retrieved_context = query_pipeline(index, job_description, top_k=30)
    
    # Save retrieved context to file to avoid PowerShell interpretation issues
    with open("retrieved_context.txt", "w", encoding="utf-8") as f:
        total_chars = sum(len(ctx) for ctx in retrieved_context)
        f.write(f"Retrieved Context: {len(retrieved_context)} chunks, {total_chars} total characters\n\n")
        for i, context in enumerate(retrieved_context):
            f.write(f"{i+1}. [{len(context)} chars]\n{context}\n\n" + "="*80 + "\n\n")
    print(f"Retrieved {len(retrieved_context)} context chunks. Saved to retrieved_context.txt")
    
    graph.set_entry_point(RESEARCH)
    graph.add_node(RESEARCH, research_node)
    graph.add_node(SUMMARIZE, summarize_node)
    graph.add_node(GENERATE, generate_node)
    graph.add_node(REFLECT, reflect_node)
    graph.add_node(HUMAN_REVISION, human_revision_node)
    
    graph.add_edge(RESEARCH, SUMMARIZE)
    graph.add_edge(SUMMARIZE, GENERATE)
    graph.add_conditional_edges(
        
        GENERATE,conditional_revision,
        {
            "continue": REFLECT,
             END: END
        }
        ) 
    
    graph.add_edge(REFLECT, HUMAN_REVISION)
    graph.add_edge(HUMAN_REVISION, GENERATE)  # loop back to generation for revisions
    
    app = graph.compile()
    print(app.get_graph().draw_mermaid())
    app.get_graph().print_ascii()
    
    
    final_state=app.invoke({
        "job_description": job_description,
        "retrieved_context": retrieved_context
    })

    
    print("\n--- Processing your Resume (Llama 3 is thinking...) ---")
   
    print(final_state["current_draft"])