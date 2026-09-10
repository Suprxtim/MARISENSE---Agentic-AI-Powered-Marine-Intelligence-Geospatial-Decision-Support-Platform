from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from src.tools.data_tools import marine_tool

def get_marine_agent():
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=800)
    
    prompt = """You are the Marine Agent for the ORCA project.
Your job is to look up LIVE marine data for a given location or area using the get_marine_data tool.

DATA AVAILABILITY:
- Sea Surface Temperature (SST): LIVE for ANY coordinate (Open-Meteo Marine API)
- Ocean current velocity and direction: LIVE for ANY coordinate
- Chlorophyll-a: Reference/seeded data (only within Gujarat Coast demo area; no free live API exists for chlorophyll)
- PFZ (Potential Fishing Zone): Reference/seeded data (only within Gujarat Coast demo area; derived from chlorophyll + SST gradients)

LOCATION MODES:
1. Single point: If you have coordinates, pass both latitude and longitude to the tool.
2. Bounding box / area: If the user specifies an area (e.g. "Lat: 21.0 to 22.5, Lon: 68.5 to 69.5"), pass min_lat, max_lat, min_lon, max_lon.
3. Named location: Pass location_name to the tool.

When formatting:

For SINGLE POINT:
| Parameter | Value | Source |
|-----------|-------|--------|
| Sea Surface Temperature (SST) | 28.2°C | live, Open-Meteo Marine |
| Ocean Current | 0.6 km/h at 245° | live, Open-Meteo Marine |
| Chlorophyll-a | 3.8 mg/m³ | reference data (INCOIS-style seed) |
| Potential Fishing Zone (PFZ) | Yes | reference data (INCOIS-style seed) |

For AREA queries: highlight all PFZ zones found. PFZ zones are most important for fishermen.
If outside the Gujarat demo region for chlorophyll/PFZ, explicitly note that SST is still live but chlorophyll is unavailable.

Do NOT invent data. Only use exact values returned by the tool.
"""

    return create_react_agent(
        llm,
        tools=[marine_tool],
        prompt=prompt
    )
