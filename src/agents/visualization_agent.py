import json
import os
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

def visualization_node(state):
    """
    Parses the conversation history to extract structured data 
    and generates GeoJSON layers for the map.
    """
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=800)
    
    # Build conversation string to feed to the LLM
    conversation_text = ""
    for msg in state.get("messages", []):
        role = "AI" if getattr(msg, "type", "") == "ai" else "User"
        content = msg.content
        if isinstance(content, list):
            content = "".join([b.get("text", "") for b in content if isinstance(b, dict) and "text" in b])
        conversation_text += f"{role}: {content}\n\n"
        
    prompt = f"""You are the Visualization Agent for the ORCA project.
Extract the location (look for 'resolved_latitude' and 'resolved_longitude' in the tool outputs), wave severity, PFZ status, SST, and Geofence status from the conversation.
Return a JSON object EXACTLY matching this schema:
{{
    "lat": 22.5,
    "lng": 69.2,
    "has_pfz": true/false,
    "has_sst": true/false,
    "sst_value": 28.5,
    "has_waves": true/false,
    "wave_severity": "green" | "yellow" | "red",
    "inside_restricted_zone": true/false,
    "zone_name": "name of zone if any"
}}
Return ONLY valid JSON. If a value is unknown, use false or null.

History:
{conversation_text}
"""
    
    response = llm.invoke([HumanMessage(content=prompt)], config={"tags": ["visualization_llm"]})
    
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
            
        data = json.loads(content.strip())
    except Exception as e:
        print("Visualization Agent JSON parse error:", e)
        return {"layers": []}
        
    layers = []
    lat = data.get("lat")
    lng = data.get("lng")
    
    if lat is None or lng is None:
        return {"layers": []}
        
    # 1. PFZ Marker
    if str(data.get("has_pfz")).lower() == "true":
        layers.append({
            "id": "pfz_marker",
            "label": "Potential Fishing Zone",
            "visible": True,
            "type": "point",
            "geojson": {"type": "Point", "coordinates": [lng, lat]},
            "style": {"color": "green", "radius": 8}
        })
        
    # 2. SST Indicator
    if str(data.get("has_sst")).lower() == "true" and data.get("sst_value") is not None:
        try:
            sst = float(data.get("sst_value", 0))
            color = "blue" if sst < 25 else ("orange" if sst < 29 else "red")
            layers.append({
                "id": "sst_indicator",
                "label": f"SST ({sst}°C)",
                "visible": True,
                "type": "point",
                "geojson": {"type": "Point", "coordinates": [lng, lat]},
                "style": {"color": color, "radius": 12}
            })
        except:
            pass
        
    # 3. Wave Hazard
    if str(data.get("has_waves")).lower() == "true":
        sev = data.get("wave_severity", "blue")
        layers.append({
            "id": "wave_hazard",
            "label": f"Wave Hazard",
            "visible": True,
            "type": "circle",
            "geojson": {"type": "Point", "coordinates": [lng, lat]},
            "style": {"color": sev, "radius": 20000} # 20km radius circle
        })
        
    # 4. Geofence Boundary
    inside = data.get("inside_restricted_zone")
    if str(inside).lower() == "true":
        # The file is at the root of the project (two levels up from src/agents)
        geojson_path = os.path.join(os.path.dirname(__file__), "..", "..", "mock_geofence_data.geojson")
        poly_geojson = None
        zone_name = data.get("zone_name", "") or ""
        
        if os.path.exists(geojson_path):
            with open(geojson_path, "r") as f:
                geo_data = json.load(f)
                for feature in geo_data.get("features", []):
                    name = feature.get("properties", {}).get("name", "")
                    if name and (name in zone_name or zone_name in name):
                        poly_geojson = feature.get("geometry")
                        break
                
                # fallback to first feature if no exact name match
                if not poly_geojson and geo_data.get("features"):
                    poly_geojson = geo_data["features"][0].get("geometry")
                    
        if poly_geojson:
            layers.append({
                "id": "geofence_boundary",
                "label": f"Restricted: {zone_name}",
                "visible": True,
                "type": "polygon",
                "geojson": poly_geojson,
                "style": {"color": "red", "weight": 2, "fillOpacity": 0.4}
            })

    # Return layers, appending them to the state (AgentState handles this if properly typed, 
    # but since layers is not Annotated with operator.add, it will just overwrite the state field, which is what we want)
    return {"layers": layers}
