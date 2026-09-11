from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from src.agents.state import AgentState

async def language_node(state: AgentState):
    """
    Translates the final English output into the selected target language.
    If the target language is English, it returns the state unmodified.
    """
    target_language = state.get("target_language", "English")
    
    if target_language.lower() == "english":
        return {"messages": []} # No change needed
        
    # Get the latest message (which should be from the risk/synthesis agent)
    latest_message = state["messages"][-1]
    # 2. LLM Translation Fallback
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=1000)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert translator. Translate the following text into {language}. Preserve all technical terms, formatting, and the tone of a marine safety advisory. Only return the translated text without any conversational filler or preambles."),
        ("user", "{text}")
    ])
    
    chain = prompt | llm
    
    result = await chain.ainvoke({
        "language": target_language,
        "text": latest_message.content
    })
    
    return {"messages": [result]}
