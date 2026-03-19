from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM

# Now this will work

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", "You are  Tier-1 Executive Resume Writer and a Technical Recruiter specializing in European Big Tech and Automotive AI."
            "Synthesize the old resume or resumes into one Master ATS-Optimized CV for user using the job description provided by user. The goal is a 90% keyword match for the provided Job Description while maintaining actual facts and a human, high-impact narrative." 
            " If the user provides critique, respond with a revised version  of your previous attempts."
        ),
        MessagesPlaceholder(variable_name="messages"),
        
        
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", "You are a Expert  Technical Recruiter with expertise in Data Science, ML Engineering, and ATS-Optimized CVs for European Big Tech and Automotive AI."
            "Review the previous resume and critique it based on the provided Job Description. Provide detailed recommendations  which are Focused on improving the resume by keyword match, keeping actual information, while ensuring the narrative remains human and impactful. Provide specific feedback on what can be improved in the next iteration." 
            
        ),
        MessagesPlaceholder(variable_name="messages"),
        
        
    ]
)

llm = OllamaLLM(model="llama3")

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

# The 'messages' here fills the MessagesPlaceholder
chain = generation_chain | reflection_chain
