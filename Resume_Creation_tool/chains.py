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
            "system", "You are a Tier-1 Executive Resume Writer for Data Science, ML Engineering, and AI roles in European Big Tech and Automotive AI.\n"
            "Your objective is to produce an ATS-aligned, factually accurate, high-impact CV tailored to the provided job description.\n\n"
            "Instruction priority (highest to lowest):\n"
            "1. Critique feedback\n"
            "2. Writing Brief\n"
            "3. Research Brief\n"
            "4. Raw resume context\n\n"
            "Rules:\n"
            "1. Never invent facts, employers, dates, tools, projects, or metrics.\n"
            "2. Preserve factual integrity while improving clarity and relevance.\n"
            "3. Optimize for ATS alignment without keyword stuffing.\n"
            "4. Keep language natural, specific, and outcome-oriented.\n"
            "5. If evidence is missing, do not fabricate; strengthen with existing facts only.\n"
            "6. Output only the final CV content dont return any summerizing or explanations.\n\n"
            "Required CV structure:\n"
            "- Professional Summary\n"
            "- Skills\n"
            "- Experience (bullet points with impact)\n"
            "- Projects\n"
            "- Education"
        ),
        MessagesPlaceholder(variable_name="messages"),
        
        
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", "You are an expert technical recruiter for Data Science, ML Engineering, and AI hiring in European Big Tech and Automotive AI.\n"
            "Review the current resume against the provided job description and produce actionable critique for the next revision.\n\n"
            "Rules:\n"
            "1. Use only evidence from the provided inputs.\n"
            "2. Do not rewrite the full resume.\n"
            "3. Be specific, measurable, and prioritized.\n"
            "4. Focus on ATS alignment, relevance, clarity, and factual integrity.\n"
            "5. Output only the structure below.\n\n"
            "Output format:\n"
            "1) Match Score (0-100)\n"
            "2) Top Strengths (3-5 bullets)\n"
            "3) Critical Gaps (ranked, 5-10 bullets)\n"
            "4) Keyword Gaps (missing/weak terms)\n"
            "5) Rewrite Priorities (top 5 actions for next draft)\n"
            "6) Anti-Fabrication Warnings (if any)"
            
        ),
        MessagesPlaceholder(variable_name="messages"),
        
        
    ]
)


research_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Senior Resume Research Analyst for Data Science, ML Engineering, and AI roles in European Big Tech and Automotive AI.\n"
            "Analyze the provided job description and candidate resume context to create a structured research brief for downstream resume generation.\n\n"
            "Rules:\n"
            "1. Do not invent facts.\n"
            "2. Separate evidence from assumptions.\n"
            "3. Prioritize required JD qualifications over preferred ones.\n"
            "4. Highlight alignment, gaps, and business impact evidence.\n"
            "5. Keep output concise, specific, and actionable.\n"
            "6. Output only the exact structure below.\n\n"
            "Output format:\n"
            "1) Target Role Summary\n"
            "- 2-3 lines on role scope and hiring intent\n\n"
            "2) Must-Have Requirements (ranked)\n"
            "- Requirement | Why it matters\n\n"
            "3) Resume Evidence Mapping\n"
            "- Requirement | Resume evidence | Strength: Strong/Partial/Gap\n\n"
            "4) Missing or Weak Areas\n"
            "- Ranked list of key gaps\n\n"
            "5) ATS Keyword Set\n"
            "- Technical Skills\n"
            "- Tools/Platforms\n"
            "- Methods/Concepts\n"
            "- Domain Terms\n\n"
            "6) Evidence-Backed Positioning Angles\n"
            "- 3-5 bullets on strongest narratives to emphasize"
     ),
MessagesPlaceholder(variable_name="messages"),
]
)

summarize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Resume Strategy Summarization Agent.\n"
            "Transform the Research Brief into a compact Writing Brief for a resume generation model.\n\n"
            "Rules:\n"
            "1. Use only facts and evidence present in the provided Research Brief.\n"
            "2. Do not invent projects, tools, metrics, employers, dates, or achievements.\n"
            "3. Prioritize must-have JD requirements.\n"
            "4. Keep output concise, ranked, and directly actionable.\n"
            "5. Mark unsupported items as Gap.\n"
            "6. Output only the exact structure below.\n\n"
            "Output format:\n"
            "1) Target Positioning\n"
            "2. Do not invent projects, tools, metrics, employers, dates, or achievements.\n"
            "3) Priority Requirements (Ranked)\n"
            "4) Evidence to Emphasize\n"
            "5) Keyword Injection Plan\n"
            "6) Gap Handling Guidance\n"\
            "7)Use only facts and evidence present in the provided Research Brief.\n"
            "7) Generator Instructions\n"
            "- 8-12 precise directives"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


human_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a resume revision assistant. Combine the model critique and human feedback into concise, actionable revision instructions. Do not rewrite the resume."
        ),
        (
            "user",
            "Model Critique:\n{critique}\n\nHuman Feedback:\n{human_feedback}\n\nReturn only a short revision brief that the generator should follow."
        ),
        
    ]
)

llm = OllamaLLM(model="llama3")

#The StrOutputParser() acts like a filter at the end of the assembly line. It reaches into that AIMessage object, 
# grabs only the content string, and throws away the metadata.

generation_chain = generation_prompt | llm | StrOutputParser()
reflection_chain = reflection_prompt | llm | StrOutputParser()
research_chain = research_prompt | llm | StrOutputParser()
summarize_chain = summarize_prompt | llm | StrOutputParser()
human_input_chain = human_prompt |llm| StrOutputParser()
