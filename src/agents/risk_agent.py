from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

def get_risk_agent():
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=800)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Synthesis and Risk Assessment Agent for the ORCA maritime intelligence system.
Read the raw structured data gathered by the Marine, Weather, and Geospatial tools, then produce a verdict with structured sections.

CRITICAL RULE: If `inside_restricted_zone` is true, you MUST declare UNSAFE regardless of weather.

Output Format — follow this EXACTLY:
VERDICT: [SAFE | CAUTION | UNSAFE]

## Sea & Weather
[1-2 sentences on wind speed, gusts, wave height, swell, and precipitation. Be specific with numbers.]

## Marine Conditions
[1-2 sentences on SST, ocean current speed, and chlorophyll/PFZ status. Note if chlorophyll is below PFZ threshold.]

## Hazards
[1 sentence on geofence/restricted zone status, cyclone alert, and lightning risk. If all clear, state that explicitly.]

## Assessment
[1-2 sentence bottom-line recommendation. What should the operator actually do?]"""),
        ("user", "Here is the data gathered so far:\n\n{context}\n\nPlease provide your final verdict.")
    ])
    
    def format_context(inputs):
        msgs = inputs["messages"]
        context = ""
        for m in msgs:
            text = ""
            if isinstance(m.content, list):
                text = "".join(b.get("text", "") for b in m.content if isinstance(b, dict))
            else:
                text = str(m.content)
            role = "User" if m.type == "human" else "Agent"
            context += f"{role}: {text}\n\n"
        return {"context": context}
        
    def wrap_in_dict(ai_message):
        return {"messages": [ai_message]}
        
    return format_context | prompt | llm | wrap_in_dict
