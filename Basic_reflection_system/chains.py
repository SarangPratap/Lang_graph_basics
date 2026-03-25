from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser


# Now this will work
# Step 1: The ChatPromptTemplate Constructor
# Think of this as an envelope designer. Instead of just 
# sending a raw block of text to Llama 3, you are creating a professional layout with specific "slots" for different types of information.

# Step 2: The "System" Message (The Persona)
# The first item in the list ("system", "You are a Tier-1...") 
# is the System Prompt.
# What it does: It sets the "Permanent Rules" for the AI.
# Why it matters for you: Since you are a Data Science student 
# targeting European Big Tech, this instruction tells Llama 3 to stop acting like a general chatbot and 
# start acting like a recruiter who knows about Automotive AI and ATS optimization.
# Behavior: It stays at the very top of every request, ensuring the AI never 
# "forgets" its role during the reflection process.


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", "You are  Tier-1 Executive Resume Writer and a Technical Recruiter specializing in European Big Tech and Automotive AI."
            "Synthesize the old resume or resumes into one Master ATS-Optimized CV for user using the job description provided by user. The goal is a 90% keyword match for the provided Job Description while maintaining actual facts and a human, high-impact narrative." 
            " If the user provides critique, respond with a revised version  of your previous attempts.Output only the CV content. No acknowledgements, no critique, no explanations"
        ),
        MessagesPlaceholder(variable_name="messages"),
        
        
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", "You are a Expert Technical Recruiter with expertise in Data Science, ML Engineering, and ATS-Optimized CVs for European Big Tech and Automotive AI who will critique the previous resume and provide feedback based on the provided Job Description. Provide detailed recommendations which are Focused on improving the resume by keyword match, keeping actual information, while ensuring the narrative remains human and impactful. Provide specific feedback on what can be improved in the next iteration." 
            "Critique : the previous resume and provide feedback based on the provided Job Description. Provide detailed recommendations which are Focused on improving the resume by keyword match, keeping actual information, while ensuring the narrative remains human and impactful. Provide specific feedback on what can be improved in the next iteration." 
            
        ),
        MessagesPlaceholder(variable_name="messages"),
        
        
    ]
)

llm = OllamaLLM(model="llama3")

#The StrOutputParser() acts like a filter at the end of the assembly line. It reaches into that AIMessage object, 
# grabs only the content string, and throws away the metadata.

generation_chain = generation_prompt | llm | StrOutputParser()
reflection_chain = reflection_prompt | llm | StrOutputParser()


