from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from src.agents.state import AgentState
from src.agents.marine_agent import get_marine_agent
from src.agents.weather_agent import get_weather_agent
from src.agents.geospatial_agent import get_geospatial_agent
from src.agents.risk_agent import get_risk_agent
from src.agents.visualization_agent import visualization_node

def create_supervisor_graph():
    """
    Compiles the multi-agent graph. 
    Flow: User Query -> Query Rewriter -> Intent Router -> Marine Agent -> Weather Agent -> Geospatial Agent -> Risk Agent -> Visualization -> Final Output
    The Query Rewriter resolves vague follow-up queries (e.g. "what about tomorrow?")
    into fully explicit standalone queries using conversation history.
    """
    builder = StateGraph(AgentState)
    
    # ── QUERY REWRITER NODE ────────────────────────────────────────────────
    async def query_rewriter_node(state: AgentState):
        """
        Takes the current user message + recent conversation history and rewrites
        vague follow-ups into fully explicit, standalone queries.
        
        e.g. History: "Is it safe at 22.5, 69.2?" -> User: "what about tomorrow?"
             Rewritten: "Is it safe at latitude 22.5, longitude 69.2 tomorrow?"
        
        If the query is already fully specified or there's no useful history,
        it passes through unchanged.
        """
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate

        current_query = state["messages"][0].content
        history = state.get("conversation_history", [])

        # If no prior history, the query must already be standalone — skip rewriting
        if not history:
            return {"messages": []}

        # Format the last few turns of conversation for the LLM
        history_text = ""
        # Keep only the last 6 turns (3 exchanges) to stay focused
        recent = history[-6:]
        for turn in recent:
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            history_text += f"{role}: {content}\n"

        llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=300)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query resolver for a maritime safety AI system called ORCA.

Your ONLY job: Given a conversation history and a new user message, determine whether the new message is a vague follow-up that relies on context from previous turns. If it is, rewrite it into a fully explicit, standalone query. If it is already complete and self-contained, return it exactly as-is.

Rules:
1. If the user refers to "there", "that location", "same place", or omits a location — fill in the most recent location (coordinates or name) from the history.
2. If the user says "tomorrow", "tonight", "next week" — keep the time reference but fill in any missing location.
3. If the user narrows scope (e.g. "just the wave height") — rewrite to ask specifically about that parameter at the most recent location.
4. If the user asks something completely new and unrelated to prior context — return it as-is without injecting old context.
5. NEVER add information the user didn't ask about. Only fill in implicit references from history.
6. Return ONLY the rewritten query text. No explanations, no quotes, no preamble."""),
            ("user", "CONVERSATION HISTORY:\n{history}\n\nNEW USER MESSAGE:\n{query}\n\nRewritten query:")
        ])

        chain = prompt | llm
        result = await chain.ainvoke({"history": history_text, "query": current_query}, config={"tags": ["internal_llm"]})

        rewritten = result.content.strip().strip('"').strip("'")

        # If the rewriter returned something meaningful, replace the original human message
        if rewritten and rewritten.lower() != current_query.lower():
            # Replace the first message (the user's query) with the rewritten version
            return {"messages": [HumanMessage(content=rewritten)]}
        
        # No rewrite needed
        return {"messages": []}

    async def _execute_tool_node(state: AgentState, tool_instance):
        from langchain_groq import ChatGroq
        from langchain_core.messages import ToolMessage, SystemMessage
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
        llm_with_tools = llm.bind_tools([tool_instance], tool_choice=tool_instance.name)
        
        system_msg = SystemMessage(content="You are a silent data extractor. You must call the provided tool immediately. Do not generate any conversational text, markdown tables, or commentary.")
        
        # Use the latest human message (which may have been rewritten by query_rewriter)
        # Find the last HumanMessage in the state
        original_query = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                original_query = msg
                break
        if original_query is None:
            original_query = state["messages"][0]
        
        messages = [system_msg, original_query]
        
        ai_msg = await llm_with_tools.ainvoke(messages)
        if ai_msg.tool_calls:
            tool_call = ai_msg.tool_calls[0]
            result = await tool_instance.ainvoke(tool_call["args"])
            tool_msg = ToolMessage(content=str(result), tool_call_id=tool_call["id"], name=tool_instance.name)
            return {"messages": [ai_msg, tool_msg]}
        return {"messages": [ai_msg]}

    async def marine_node(state: AgentState):
        from src.tools.data_tools import marine_tool
        return await _execute_tool_node(state, marine_tool)
        
    async def weather_node(state: AgentState):
        from src.tools.data_tools import weather_tool
        return await _execute_tool_node(state, weather_tool)
        
    async def geospatial_node(state: AgentState):
        from src.agents.geospatial_agent import check_geofence
        return await _execute_tool_node(state, check_geofence)
        
    async def risk_node(state: AgentState):
        agent = get_risk_agent()
        result = await agent.ainvoke({"messages": state["messages"]})
        return {"messages": [result["messages"][-1]]}
        
    async def vis_node_wrapper(state: AgentState):
        return visualization_node(state)
        
    async def language_node(state: AgentState):
        from src.agents.language_agent import language_node as ln
        return await ln(state)

    async def route_intent(state: AgentState):
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        
        # Use the latest HumanMessage (potentially rewritten)
        query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break
        if not query:
            query = state["messages"][0].content

        llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an intent router for a marine and fishing safety AI. Analyze the user's query.\n"
                       "If the user is asking about fishing safety, marine conditions, weather, locations, or geofencing, reply EXACTLY with 'data_query'.\n"
                       "If the user is saying hello, making small talk, reporting errors, or asking general non-location questions, reply EXACTLY with 'conversational'."),
            ("user", "{query}")
        ])
        
        chain = prompt | llm
        result = await chain.ainvoke({"query": query}, config={"tags": ["internal_llm"]})
        intent = result.content.strip().lower()
        
        if "data_query" in intent:
            return "marine"
        return "conversational"

    async def conversational_node(state: AgentState):
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        
        query = state["messages"][-1].content
        target_language = state.get("target_language", "English")
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are ORCA AI, a helpful and friendly maritime assistant. Respond naturally to the user. Keep it concise. If they report an error, be empathetic. IMPORTANT: Always respond entirely in {target_language}."),
            ("user", "{query}")
        ])
        
        chain = prompt | llm
        result = await chain.ainvoke({"query": query, "target_language": target_language})
        
        return {"messages": [result]}
        
    # Add nodes to graph
    builder.add_node("query_rewriter", query_rewriter_node)
    builder.add_node("marine", marine_node)
    builder.add_node("weather", weather_node)
    builder.add_node("geospatial", geospatial_node)
    builder.add_node("risk", risk_node)
    builder.add_node("conversational", conversational_node)
    builder.add_node("visualization", vis_node_wrapper)
    
    # Define the sequential orchestration flow
    # Query Rewriter runs first, then intent routing decides the path
    builder.add_edge(START, "query_rewriter")
    builder.add_conditional_edges("query_rewriter", route_intent, {"marine": "marine", "conversational": "conversational"})
    builder.add_edge("marine", "weather")
    builder.add_edge("weather", "geospatial")
    builder.add_edge("geospatial", "risk")
    builder.add_edge("risk", "visualization")
    builder.add_edge("conversational", "visualization")
    builder.add_edge("visualization", END)
    
    return builder.compile()

