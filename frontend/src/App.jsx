import { useState, useRef, useEffect, useCallback } from "react";
import { MapContainer, TileLayer, Marker, Popup, Circle, GeoJSON, CircleMarker, Rectangle, Polyline, useMapEvents, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { ShieldAlert, Ship, Activity, Send, Loader2, Layers, PenTool, X, MapPin, Wind, Thermometer, Fish, Route, CheckCircle, AlertTriangle, XCircle, Sprout, Brain, User, Bot, Navigation, Mic, MicOff, Volume2, VolumeX } from "lucide-react";
import "./index.css";
import L from "leaflet";
import icon from "leaflet/dist/images/marker-icon.png";
import iconShadow from "leaflet/dist/images/marker-shadow.png";
L.Marker.prototype.options.icon = L.icon({ iconUrl: icon, shadowUrl: iconShadow, iconAnchor: [12, 41] });

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function chlToColor(chl) {
  if (chl === null || chl === undefined) return null;
  if (chl >= 2.0) return "rgba(0, 255, 80, 0.6)";
  if (chl >= 1.0) return "rgba(80, 220, 0, 0.5)";
  if (chl >= 0.5) return "rgba(200, 230, 0, 0.38)";
  return "rgba(0, 150, 200, 0.25)";
}

function verdictStyle(verdict) {
  const v = (verdict || "").toUpperCase();
  if (v.includes("SAFE") && !v.includes("UN")) return { color: "#69f0ae", bg: "rgba(105,240,174,0.12)", Icon: CheckCircle };
  if (v.includes("UNSAFE") || v.includes("DANGER")) return { color: "#ff5252", bg: "rgba(255,82,82,0.12)", Icon: XCircle };
  return { color: "#ffd740", bg: "rgba(255,215,64,0.12)", Icon: AlertTriangle };
}

// Parse VERDICT / structured sections from agent response
function parseAgentResponse(text) {
  if (!text) return null;
  const vMatch = text.match(/VERDICT\s*[:\-]?\s*([A-Z]+)/i);
  const verdict = vMatch ? vMatch[1].trim().toUpperCase() : null;

  // Try to parse structured sections (## SEA & WEATHER, ## MARINE, etc.)
  const sections = [];
  const sectionRegex = /##\s*([^\n]+)\n([\s\S]*?)(?=\n##|$)/g;
  let m;
  while ((m = sectionRegex.exec(text)) !== null) {
    sections.push({ title: m[1].trim(), body: m[2].trim() });
  }

  // Fallback: just grab reasoning paragraph
  let reasoning = text;
  const rMatch = text.match(/REASONING\s*[:\-]?\s*([\s\S]+)/i);
  if (rMatch) reasoning = rMatch[1].trim();

  return { verdict, sections, reasoning, hasSections: sections.length > 0 };
}

// Friendly label mapping for Intel Feed fields
const FIELD_LABELS = {
  sst_celsius: { label: "Sea Surface Temp", unit: "°C" },
  ocean_current_velocity_kmh: { label: "Current Speed", unit: " km/h" },
  ocean_current_direction_deg: { label: "Current Direction", unit: "°" },
  wind_speed_knots: { label: "Wind Speed", unit: " kn" },
  wind_gusts_knots: { label: "Wind Gusts", unit: " kn" },
  wave_height_meters: { label: "Wave Height", unit: " m" },
  swell_height_meters: { label: "Swell Height", unit: " m" },
  wave_period_seconds: { label: "Wave Period", unit: " s" },
  chlorophyll_mg_m3: { label: "Chlorophyll-a", unit: " mg/m³" },
  phytoplankton_mmol_m3: { label: "Phytoplankton", unit: " mmol/m³" },
  is_pfz: { label: "Fishing Zone (PFZ)", unit: "" },
  cyclone_alert: { label: "Cyclone Alert", unit: "" },
  lightning_risk: { label: "Lightning Risk", unit: "" },
  weather_description: { label: "Weather", unit: "" },
  precipitation_mm: { label: "Precipitation", unit: " mm" },
  inside_restricted_zone: { label: "Restricted Zone", unit: "" },
  distance_to_nearest_boundary_km: { label: "Distance to Boundary", unit: " km" },
  resolved_latitude: { label: "Latitude", unit: "" },
  resolved_longitude: { label: "Longitude", unit: "" },
  center_sst_celsius: { label: "Area SST (center)", unit: "°C" },
  center_ocean_current_kmh: { label: "Area Current", unit: " km/h" },
  total_pfz_found: { label: "PFZ Zones Found", unit: "" },
};

// Fields to always skip in Intel Feed
const SKIP_FIELDS = new Set([
  "matched_location","distance_to_reference_km","copernicus_grid_point",
  "query_type","bounding_box","coverage_note","pfz_source","chlorophyll_source",
  "sst_source","pfz_confidence","pfz_classification_confidence","pfz_low_confidence",
  "chlorophyll_low_confidence","note","copernicus_distance_km",
  "locations_in_area","locations_found","total_locations_found",
  "other_locations_in_area","pfz_zones_in_area","total_copernicus_points",
]);

function formatFieldValue(key, value) {
  if (value === null || value === undefined || value === "null" || value === "") return null;
  if (typeof value === "boolean") return value ? "Yes" : "No";
  const str = String(value);
  if (str === "0" && (key.includes("found") || key.includes("total"))) return null; // hide zero counts
  return str;
}

// ─── Components ───────────────────────────────────────────────────────────────

function MetricCard({ Icon, label, value, unit, color }) {
  return (
    <div style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 8, padding: "8px 12px", display: "flex", flexDirection: "column", gap: 2, flex: "1 1 110px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: 0.4 }}>
        {Icon && <Icon size={11} />} {label}
      </div>
      <div style={{ fontWeight: 700, fontSize: "0.95rem", color: color || "var(--text-primary)" }}>
        {(value !== null && value !== undefined && value !== "") ? `${value}${unit || ""}` : "—"}
      </div>
    </div>
  );
}

function StructuredResponse({ msg }) {
  if (msg.isError) {
    return <div style={{ fontSize: "0.9rem", color: "var(--alert-red)" }}>{msg.text}</div>;
  }

  const parsed = parseAgentResponse(msg.text);
  if (!parsed || !parsed.verdict) {
    return <div style={{ fontSize: "0.9rem", color: "var(--text-primary)", whiteSpace: "pre-wrap", lineHeight: 1.6 }}>{msg.text}</div>;
  }

  const vs = verdictStyle(parsed.verdict);
  const VIcon = vs.Icon;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Verdict badge */}
      <div style={{ display: "inline-flex", alignItems: "center", gap: 8, background: vs.bg, border: `1.5px solid ${vs.color}`, borderRadius: 8, padding: "6px 14px", alignSelf: "flex-start" }}>
        <VIcon size={16} color={vs.color} />
        <span style={{ fontWeight: 800, fontSize: "0.92rem", color: vs.color, letterSpacing: 1.2 }}>{parsed.verdict}</span>
      </div>

      {/* Structured sections if present */}
      {parsed.hasSections ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {parsed.sections.map((s, i) => (
            <div key={i} style={{ borderLeft: `2px solid ${vs.color}30`, paddingLeft: 10 }}>
              <div style={{ fontSize: "0.72rem", fontWeight: 700, color: vs.color, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 4 }}>{s.title}</div>
              <div style={{ fontSize: "0.87rem", color: "var(--text-primary)", lineHeight: 1.65, whiteSpace: "pre-wrap" }}>{s.body}</div>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ fontSize: "0.87rem", color: "var(--text-primary)", lineHeight: 1.75, whiteSpace: "pre-wrap" }}>{parsed.reasoning}</div>
      )}
    </div>
  );
}

// ─── Map Handlers ─────────────────────────────────────────────────────────────

function DrawAreaHandler({ isDrawing, onDrawComplete }) {
  const [startPoint, setStartPoint] = useState(null);
  const [currentRect, setCurrentRect] = useState(null);
  const map = useMap();
  useEffect(() => {
    if (!isDrawing) { setStartPoint(null); setCurrentRect(null); map.dragging.enable(); map.getContainer().style.cursor = ""; return; }
    map.dragging.disable(); map.getContainer().style.cursor = "crosshair";
    return () => { map.dragging.enable(); map.getContainer().style.cursor = ""; };
  }, [isDrawing, map]);
  useMapEvents({
    mousedown(e) { if (!isDrawing) return; setStartPoint(e.latlng); },
    mousemove(e) { if (!isDrawing || !startPoint) return; setCurrentRect(L.latLngBounds(startPoint, e.latlng)); },
    mouseup(e) {
      if (!isDrawing || !startPoint) return;
      const b = L.latLngBounds(startPoint, e.latlng);
      const sw = b.getSouthWest(), ne = b.getNorthEast();
      if (Math.abs(ne.lat - sw.lat) > 0.01 && Math.abs(ne.lng - sw.lng) > 0.01) onDrawComplete(b);
      setStartPoint(null); setCurrentRect(null);
    }
  });
  return currentRect ? <Rectangle bounds={currentRect} pathOptions={{ color: "#00e5ff", weight: 2, fillColor: "#00e5ff", fillOpacity: 0.15, dashArray: "6 4" }} /> : null;
}

// Route handler: click to add points, uses a "Finish Route" button (no dblclick)
function RouteDrawHandler({ isRouting, routePoints, onAddPoint }) {
  const map = useMap();
  useEffect(() => {
    if (!isRouting) { map.getContainer().style.cursor = ""; return; }
    map.getContainer().style.cursor = "crosshair";
    return () => { map.getContainer().style.cursor = ""; };
  }, [isRouting, map]);
  useMapEvents({
    click(e) { if (!isRouting) return; onAddPoint(e.latlng); }
  });
  return routePoints.length >= 2 ? (
    <Polyline positions={routePoints} pathOptions={{ color: "#ffd740", weight: 3, dashArray: "none" }} />
  ) : null;
}

function MapClickHandler({ onMapClick, isDrawing, isRouting }) {
  useMapEvents({ click(e) { if (isDrawing || isRouting) return; onMapClick(e.latlng.lat, e.latlng.lng); } });
  return null;
}

// ─── Static markers ───────────────────────────────────────────────────────────
const MARKERS = [
  { pos: [22.0, 69.5],  label: "Harbor Point" },
  { pos: [21.5, 69.0],  label: "Zone A Nearshore" },
  { pos: [21.0, 68.5],  label: "Zone B Offshore" },
  { pos: [22.6, 69.8],  label: "Gulf Kutch Inner" },
  { pos: [22.4, 69.1],  label: "Gulf Kutch Outer" },
  { pos: [21.6, 69.6],  label: "Porbandar Coast" },
  { pos: [22.2, 68.9],  label: "Dwarka Offshore" },
  { pos: [18.9, 72.8],  label: "Mumbai Coast" },
  { pos: [15.5, 73.7],  label: "Goa Coast" },
  { pos: [9.9,  76.2],  label: "Kochi Coast" },
  { pos: [8.0,  77.5],  label: "Kanyakumari" },
  { pos: [13.0, 80.3],  label: "Chennai Coast" },
  { pos: [17.7, 83.3],  label: "Vizag Coast" },
  { pos: [20.2, 86.7],  label: "Paradip Coast" },
  { pos: [21.6, 88.1],  label: "Kolkata Offshore" },
];

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [query, setQuery]               = useState("");
  const [chatHistory, setChatHistory]   = useState([{ role: "agent", text: "ORCA Maritime AI initialized.\n\nTip: Click any marker or ocean point to query it. Use the toolbar buttons for area selection, PFZ overlay, or route planning." }]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [reasoningTrace, setReasoningTrace] = useState([]);
  const [mapCenter, setMapCenter]       = useState([15.0, 78.0]);
  const [mapLayers, setMapLayers]       = useState([]);
  const [language, setLanguage]         = useState("English");
  const [clickedCoordinate, setClickedCoordinate] = useState(null);
  const [isDrawingArea, setIsDrawingArea]   = useState(false);
  const [isRoutingMode, setIsRoutingMode]   = useState(false);
  const [drawnBounds, setDrawnBounds]       = useState(null);
  const [safetyRing, setSafetyRing]         = useState(null);
  const [chlGrid, setChlGrid]               = useState([]);
  const [pfzLayer, setPfzLayer]             = useState([]);
  const [showChlHeatmap, setShowChlHeatmap] = useState(false);
  const [showPfzLayer, setShowPfzLayer]     = useState(false);
  const [routePoints, setRoutePoints]       = useState([]);
  const [liveMetrics, setLiveMetrics]       = useState({});
  const [isRecording, setIsRecording] = useState(false);
  const [isLocating, setIsLocating] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const chatEndRef = useRef(null);
  const currentAudioRef = useRef(null);
  // Stable session ID for multi-turn conversation context
  const sessionIdRef = useRef(`session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`);

  useEffect(() => {
    if (!voiceEnabled) {
      window.speechSynthesis.cancel();
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current = null;
      }
    }
  }, [voiceEnabled]);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/map-data/chlorophyll`)
      .then(r => r.json())
      .then(data => { setChlGrid(data.grid || []); setPfzLayer((data.grid || []).filter(p => p.is_pfz)); })
      .catch(() => {});
  }, []);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [chatHistory]);

  const handleMapClick = useCallback((lat, lng) => {
    if (isRoutingMode) return;
    setClickedCoordinate([lat, lng]); setDrawnBounds(null);
    setQuery(`Is it safe near ${lat.toFixed(4)}, ${lng.toFixed(4)}?`);
  }, [isRoutingMode]);

  const handleDrawComplete = useCallback((bounds) => {
    const sw = bounds.getSouthWest(), ne = bounds.getNorthEast();
    setDrawnBounds(bounds); setClickedCoordinate(null); setIsDrawingArea(false);
    setQuery(`Check fishing zones and safety for the selected area (Lat: ${sw.lat.toFixed(4)} to ${ne.lat.toFixed(4)}, Lon: ${sw.lng.toFixed(4)} to ${ne.lng.toFixed(4)})`);
  }, []);

  // Route: clicking the map adds a point via this callback
  const handleAddRoutePoint = useCallback((latlng) => {
    setRoutePoints(prev => [...prev, latlng]);
  }, []);

  // Finish route: called from the "Finish Route" button
  const handleFinishRoute = useCallback(() => {
    if (routePoints.length < 2) return;
    setIsRoutingMode(false);
    // Sample 5 evenly spaced points along the route for assessment
    const pts = routePoints;
    const step = Math.max(1, Math.floor((pts.length - 1) / 4));
    const sampled = [];
    for (let i = 0; i < pts.length; i += step) sampled.push(pts[i]);
    if (sampled[sampled.length - 1] !== pts[pts.length - 1]) sampled.push(pts[pts.length - 1]);
    const coordStr = sampled.map(p => `(${p.lat.toFixed(3)}, ${p.lng.toFixed(3)})`).join(" to ");
    setQuery(`Assess maritime safety along this route: ${coordStr}`);
  }, [routePoints]);

  const toggleLayer = (id) => setMapLayers(prev => prev.map(l => l.id === id ? { ...l, visible: !l.visible } : l));

  const handleLocateMe = () => {
    if ("geolocation" in navigator) {
      setIsLocating(true);
      navigator.geolocation.getCurrentPosition((position) => {
        const lat = position.coords.latitude.toFixed(4);
        const lng = position.coords.longitude.toFixed(4);
        setClickedCoordinate([lat, lng]);
        const q = `Assess maritime safety at my current location: ${lat}, ${lng}`;
        setQuery(q);
        setIsLocating(false);
        // Optionally auto-send: handleSend(q);
      }, (error) => {
        setIsLocating(false);
        alert("Unable to retrieve your location. Please ensure location permissions are granted.");
      });
    } else {
      alert("Geolocation is not supported by your browser.");
    }
  };

  const playTTS = async (textToSpeak) => {
    // Strip markdown formatting for cleaner speech
    const cleanText = textToSpeak.replace(/#+\s*/g, '').replace(/\*\*/g, '');
    
    const synth = window.speechSynthesis;
    const voices = synth.getVoices();
    const langMap = { "Hindi": "hi-IN", "Gujarati": "gu-IN", "Tamil": "ta-IN", "Bengali": "bn-IN", "English": "en-IN" };
    const targetLang = langMap[language] || "en-IN";
    
    const nativeVoice = voices.find(v => v.lang.startsWith(targetLang) || v.lang.startsWith(targetLang.split('-')[0]));
    
    // Stop any existing speech before starting a new one
    synth.cancel();
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }

    if (nativeVoice) {
      const utterance = new SpeechSynthesisUtterance(cleanText);
      utterance.voice = nativeVoice;
      utterance.lang = targetLang;
      synth.speak(utterance);
    } else {
      try {
        const res = await fetch(`${API_BASE_URL}/api/tts`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: cleanText, language })
        });
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        currentAudioRef.current = audio;
        audio.play();
      } catch (e) {
        console.error("TTS Fallback failed", e);
      }
    }
  };

  const toggleRecording = async () => {
    if (isRecording) {
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorderRef.current = new MediaRecorder(stream);
        audioChunksRef.current = [];
        mediaRecorderRef.current.ondataavailable = e => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
        mediaRecorderRef.current.onstop = async () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
          stream.getTracks().forEach(t => t.stop());
          
          setIsProcessing(true);
          setReasoningTrace([{ status: "in-progress", agent: "Transcriber", detail: "Translating voice to text..." }]);
          const formData = new FormData();
          formData.append("file", audioBlob, "voice.webm");
          formData.append("language", language);
          try {
            const res = await fetch(`${API_BASE_URL}/api/transcribe`, { method: "POST", body: formData });
            const data = await res.json();
            if (data.text) {
              setQuery(data.text);
              handleSend(data.text);
            }
          } catch(e) {
            setChatHistory(prev => [...prev, { role: "agent", text: "Voice transcription failed.", isError: true }]);
          } finally {
            setIsProcessing(false);
          }
        };
        mediaRecorderRef.current.start();
        setIsRecording(true);
      } catch (e) {
        alert("Microphone access denied or unavailable.");
      }
    }
  };

  const handleSend = async (qOverride) => {
    const userQuery = typeof qOverride === 'string' ? qOverride : query;
    if (!userQuery.trim() || isProcessing) return;
    setChatHistory(prev => [...prev, { role: "user", text: userQuery }]);
    setQuery(""); setIsProcessing(true); setLiveMetrics({});
    setReasoningTrace([{ status: "in-progress", agent: "Orchestrator", detail: "Analyzing intent and assigning agents..." }]);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userQuery, target_language: language, session_id: sessionIdRef.current })
      });
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let finalText = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        for (const line of decoder.decode(value).split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const dataStr = line.substring(6);
          if (dataStr === "[DONE]") { setIsProcessing(false); break; }
          try {
            const data = JSON.parse(dataStr);
            if (data.error) { setChatHistory(prev => [...prev, { role: "agent", text: `Error: ${data.error}`, isError: true }]); break; }

            if (data.event === "on_chat_model_stream" && data.content) finalText += data.content;

            if (data.event === "on_tool_end") {
              let detail = `Finished ${data.name}.`;
              const newMetrics = {};
              try {
                const raw = data.output?.content || data.output;
                const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
                if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
                  const lines = [];
                  for (const [k, v] of Object.entries(parsed)) {
                    if (SKIP_FIELDS.has(k) || k.endsWith("_source") || k.endsWith("_low_confidence")) continue;
                    const displayVal = formatFieldValue(k, v);
                    if (displayVal === null) continue;
                    const meta = FIELD_LABELS[k];
                    const label = meta ? meta.label : k.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
                    const unit = meta ? meta.unit : "";
                    lines.push(`${label}: ${displayVal}${unit}`);
                    newMetrics[k] = displayVal;
                  }
                  if (lines.length) detail = lines.join("\n");
                }
              } catch {}
              setLiveMetrics(prev => ({ ...prev, ...newMetrics }));
              setReasoningTrace(prev => [
                ...prev.map(t => t.status === "in-progress" ? { ...t, status: "done" } : t),
                { status: "done", agent: data.name, detail }
              ]);
            }

            if (data.event === "on_layers_ready") {
              setMapLayers(data.layers);
              const pt = data.layers.find(l => l.type === "point" || l.type === "circle");
              if (pt) setMapCenter([pt.geojson.coordinates[1], pt.geojson.coordinates[0]]);
            }
          } catch {}
        }
      }

      if (finalText) {
        setChatHistory(prev => [...prev, { role: "agent", text: finalText }]);
        setReasoningTrace(prev => [...prev, { status: "done", agent: "Risk Agent", detail: "Verdict reached." }]);
        
        if (voiceEnabled) {
          playTTS(finalText);
        }
        
        const parsed = parseAgentResponse(finalText);
        if (parsed?.verdict && clickedCoordinate) {
          setSafetyRing({ lat: clickedCoordinate[0], lng: clickedCoordinate[1], verdict: parsed.verdict });
        }
      }
    } catch {
      setChatHistory(prev => [...prev, { role: "agent", text: "Connection to ORCA failed.", isError: true }]);
    } finally { setIsProcessing(false); }
  };

  return (
    <div className="app-container">

      {/* ── LEFT: CHAT ── */}
      <div className="glass-panel chat-column">
        <div className="panel-header" style={{ justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}><Ship size={20} /> ORCA Terminal</div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button 
              onClick={() => {
                setVoiceEnabled(!voiceEnabled);
                if (voiceEnabled) window.speechSynthesis.cancel();
              }} 
              style={{ background: "transparent", border: "none", color: voiceEnabled ? "var(--accent)" : "var(--text-secondary)", cursor: "pointer", display: "flex", alignItems: "center" }}
              title="Toggle Voice Output"
            >
              {voiceEnabled ? <Volume2 size={18} /> : <VolumeX size={18} />}
            </button>
            <select value={language} onChange={e => setLanguage(e.target.value)} style={{ background: "rgba(0,0,0,0.4)", color: "var(--text-primary)", border: "1px solid var(--panel-border)", borderRadius: 4, padding: "2px 8px", fontSize: "0.85rem" }}>
              <option value="English">English</option>
              <option value="Hindi">Hindi</option>
              <option value="Gujarati">Gujarati</option>
              <option value="Tamil">Tamil</option>
              <option value="Bengali">Bengali</option>
            </select>
          </div>
        </div>

        <div style={{ flex: 1, padding: "1rem", overflowY: "auto", display: "flex", flexDirection: "column", gap: "1rem" }}>
          {chatHistory.map((msg, i) => (
            <div key={i} style={{
              alignSelf: msg.role === "user" ? "flex-end" : "flex-start",
              background: msg.role === "user" ? "rgba(0,200,255,0.12)" : (msg.isError ? "var(--alert-bg)" : "rgba(16,30,60,0.85)"),
              border: `1px solid ${msg.role === "user" ? "var(--accent)" : (msg.isError ? "var(--alert-red)" : "var(--panel-border)")}`,
              padding: "0.75rem 1rem", borderRadius: 10, maxWidth: "92%"
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: "0.7rem", color: "var(--text-secondary)", marginBottom: 6, fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.5 }}>
                {msg.role === "user" ? <User size={11} /> : <Bot size={11} />}
                {msg.role === "user" ? "You" : "ORCA AI"}
              </div>
              <StructuredResponse msg={msg} />
            </div>
          ))}
          {isProcessing && <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--accent)" }}><Loader2 className="animate-spin" size={16} /> Processing...</div>}
          <div ref={chatEndRef} />
        </div>

        <div style={{ padding: "1rem", borderTop: "1px solid var(--panel-border)", display: "flex", gap: "0.5rem" }}>
          <input value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === "Enter" && handleSend()}
            placeholder="Click the map or type a query..."
            style={{ flex: 1, background: "rgba(0,0,0,0.3)", border: "1px solid var(--panel-border)", color: "white", padding: "0.75rem", borderRadius: 6, outline: "none" }}
          />
          <button onClick={handleLocateMe} disabled={isLocating} style={{ background: "rgba(0,0,0,0.3)", color: "var(--text-secondary)", border: "1px solid var(--panel-border)", borderRadius: 6, padding: "0 10px", cursor: isLocating ? "wait" : "pointer" }} title="Use My Location">
            {isLocating ? <Loader2 size={18} className="animate-spin" /> : <MapPin size={18} />}
          </button>
          <button onClick={toggleRecording} style={{ background: isRecording ? "var(--alert-red)" : "rgba(0,0,0,0.3)", color: isRecording ? "white" : "var(--text-secondary)", border: "1px solid var(--panel-border)", borderRadius: 6, padding: "0 10px", cursor: "pointer" }} title="Record Voice">
            {isRecording ? <Mic size={18} /> : <MicOff size={18} />}
          </button>
          <button onClick={handleSend} style={{ background: "var(--accent)", color: "var(--bg-dark)", border: "none", borderRadius: 6, padding: "0 1rem", cursor: "pointer", fontWeight: 700 }}>
            <Send size={18} />
          </button>
        </div>
      </div>

      {/* ── CENTER: MAP ── */}
      <div className="glass-panel map-column">
        <div className="panel-header" style={{ justifyContent: "space-between", flexWrap: "wrap", gap: 6 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}><Activity size={20} /> Live Maritime Map</div>
          <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
            <button onClick={() => setShowChlHeatmap(v => !v)} className={`map-tool-btn${showChlHeatmap ? " active-green" : ""}`}>
              <Sprout size={12} /> Chl
            </button>
            <button onClick={() => setShowPfzLayer(v => !v)} className={`map-tool-btn${showPfzLayer ? " active-green" : ""}`}>
              <Fish size={12} /> PFZ
            </button>
            <button onClick={() => { setIsRoutingMode(v => !v); setIsDrawingArea(false); if (isRoutingMode) setRoutePoints([]); }} className={`map-tool-btn${isRoutingMode ? " active-yellow" : ""}`}>
              <Route size={12} /> Route
            </button>
            <button onClick={() => { setIsDrawingArea(v => !v); setIsRoutingMode(false); if (isDrawingArea) setDrawnBounds(null); }} className={`map-tool-btn${isDrawingArea ? " active-red" : ""}`}>
              {isDrawingArea ? <><X size={12} /> Cancel</> : <><PenTool size={12} /> Area</>}
            </button>
          </div>
        </div>

        {isDrawingArea && <div className="draw-instruction-banner"><MapPin size={13} /> Click and drag to select an area. Release to confirm.</div>}
        {isRoutingMode && (
          <div className="draw-instruction-banner" style={{ background: "rgba(255,215,64,0.08)", borderColor: "rgba(255,215,64,0.3)", color: "#ffd740", justifyContent: "space-between" }}>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <Navigation size={13} />
              {routePoints.length === 0 ? "Click on the map to add route waypoints." : `${routePoints.length} waypoint(s) added. Add more or finish.`}
            </span>
            {routePoints.length >= 2 && (
              <button onClick={handleFinishRoute} style={{ background: "#ffd740", color: "#0a1128", border: "none", borderRadius: 5, padding: "3px 10px", fontSize: "0.78rem", fontWeight: 700, cursor: "pointer" }}>
                Assess Route
              </button>
            )}
          </div>
        )}

        <div style={{ flex: 1, position: "relative" }}>
          {/* Layer legend */}
          <div style={{ position: "absolute", top: 10, right: 10, zIndex: 1000, background: "rgba(10,17,40,0.92)", padding: "10px 14px", borderRadius: 8, border: "1px solid var(--panel-border)", fontSize: "0.78rem", minWidth: 145 }}>
            <div style={{ fontWeight: 700, marginBottom: 6, color: "var(--accent)", display: "flex", alignItems: "center", gap: 5 }}><Layers size={13} /> Layers</div>
            {showChlHeatmap && (
              <>
                <div style={{ color: "var(--text-secondary)", marginBottom: 5, display: "flex", alignItems: "center", gap: 4 }}><Sprout size={11} /> Chlorophyll</div>
                {[["2.0+", "#00ff50"], ["1.0+", "#50dc00"], ["0.5+", "#c8e600"], ["< 0.5", "#0096c8"]].map(([l, c]) => (
                  <div key={l} style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 2 }}>
                    <div style={{ width: 10, height: 10, borderRadius: 2, background: c, flexShrink: 0 }} />
                    <span>{l} mg/m³</span>
                  </div>
                ))}
              </>
            )}
            {showPfzLayer && <div style={{ color: "#69f0ae", marginTop: 4, display: "flex", alignItems: "center", gap: 4 }}><Fish size={11} /> PFZ zones active</div>}
            {mapLayers.map(layer => (
              <div key={layer.id} style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 6 }}>
                <input type="checkbox" checked={layer.visible} onChange={() => toggleLayer(layer.id)} style={{ cursor: "pointer" }} />
                <span style={{ color: layer.style?.color || "white" }}>{layer.label}</span>
              </div>
            ))}
            {!showChlHeatmap && !showPfzLayer && mapLayers.length === 0 && <div style={{ color: "gray" }}>No active layers</div>}
          </div>

          <MapContainer center={mapCenter} zoom={5} style={{ height: "100%", width: "100%", background: "#0a1128", zIndex: 1 }}>
            <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" attribution="&copy; <a href=&quot;https://www.openstreetmap.org/copyright&quot;>OpenStreetMap</a> contributors &copy; <a href=&quot;https://carto.com/attributions&quot;>CARTO</a>" />

            {showChlHeatmap && chlGrid.map((pt, i) => {
              const color = chlToColor(pt.chl);
              return color ? <Circle key={i} center={[pt.lat, pt.lon]} radius={13000} pathOptions={{ color: "transparent", fillColor: color, fillOpacity: 0.75, interactive: false }} /> : null;
            })}

            {showPfzLayer && pfzLayer.map((pt, i) => (
              <CircleMarker key={i} center={[pt.lat, pt.lon]} radius={7} pathOptions={{ color: "#69f0ae", fillColor: "#69f0ae", fillOpacity: 0.85 }}>
                <Popup><strong>Potential Fishing Zone</strong><br />Chlorophyll: {pt.chl} mg/m³<br />({pt.lat}, {pt.lon})</Popup>
              </CircleMarker>
            ))}

            {safetyRing && (() => {
              const vs = verdictStyle(safetyRing.verdict);
              return <Circle center={[safetyRing.lat, safetyRing.lng]} radius={28000} pathOptions={{ color: vs.color, weight: 2, fillColor: vs.color, fillOpacity: 0.1 }}><Popup>{safetyRing.verdict}</Popup></Circle>;
            })()}

            <Rectangle bounds={[[6.0, 68.0], [24.0, 89.0]]} pathOptions={{ color: "rgba(255,255,255,0.12)", weight: 1, fillOpacity: 0, dashArray: "4 4", interactive: false }} />

            {MARKERS.map(({ pos, label }) => (
              <CircleMarker key={label} center={pos} radius={5} pathOptions={{ color: "#8ab4f8", fillColor: "#8ab4f8", fillOpacity: 0.65 }} eventHandlers={{ click: () => handleMapClick(pos[0], pos[1]) }}>
                <Popup>{label}</Popup>
              </CircleMarker>
            ))}

            {mapLayers.filter(l => l.visible).map(layer => {
              if (layer.type === "point") { const [lng, lat] = layer.geojson.coordinates; return <Circle key={layer.id} center={[lat, lng]} radius={8000} pathOptions={{ color: layer.style?.color || "cyan", fillColor: layer.style?.color, fillOpacity: 0.5 }}><Popup>{layer.label}</Popup></Circle>; }
              if (layer.type === "circle") { const [lng, lat] = layer.geojson.coordinates; return <Circle key={layer.id} center={[lat, lng]} radius={layer.style?.radius || 20000} pathOptions={{ color: layer.style?.color || "yellow", fillOpacity: 0.2 }}><Popup>{layer.label}</Popup></Circle>; }
              if (layer.type === "polygon") return <GeoJSON key={layer.id} data={layer.geojson} style={layer.style} />;
              return null;
            })}

            {routePoints.length >= 2 && <Polyline positions={routePoints} pathOptions={{ color: "#ffd740", weight: 3 }} />}
            {routePoints.map((pt, i) => (
              <CircleMarker key={i} center={pt} radius={5} pathOptions={{ color: "#ffd740", fillColor: "#ffd740", fillOpacity: 1 }}>
                <Popup>Waypoint {i + 1}</Popup>
              </CircleMarker>
            ))}

            {clickedCoordinate && <Marker position={clickedCoordinate}><Popup>Selected Location</Popup></Marker>}
            {drawnBounds && <Rectangle bounds={drawnBounds} pathOptions={{ color: "#00e5ff", weight: 2, fillColor: "#00e5ff", fillOpacity: 0.1, dashArray: "8 4" }}><Popup>Selected Area</Popup></Rectangle>}

            <MapClickHandler onMapClick={handleMapClick} isDrawing={isDrawingArea} isRouting={isRoutingMode} />
            <DrawAreaHandler isDrawing={isDrawingArea} onDrawComplete={handleDrawComplete} />
            <RouteDrawHandler isRouting={isRoutingMode} routePoints={routePoints} onAddPoint={handleAddRoutePoint} />
          </MapContainer>
        </div>
      </div>

      {/* ── RIGHT: INTEL ── */}
      <div className="glass-panel reasoning-column">
        <div className="panel-header"><ShieldAlert size={20} /> Intel Feed</div>
        <div style={{ flex: 1, padding: "1rem", overflowY: "auto", display: "flex", flexDirection: "column", gap: "0.75rem" }}>

          {Object.keys(liveMetrics).length > 0 && (
            <div>
              <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5, display: "flex", alignItems: "center", gap: 5 }}>
                <Activity size={11} /> Live Readings
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {liveMetrics["sst_celsius"]           && <MetricCard Icon={Thermometer} label="Sea Surface Temp" value={liveMetrics["sst_celsius"]} unit="°C" color="#f87171" />}
                {liveMetrics["wind_speed_knots"]       && <MetricCard Icon={Wind}       label="Wind Speed" value={liveMetrics["wind_speed_knots"]} unit=" kn" />}
                {liveMetrics["wave_height_meters"]     && <MetricCard Icon={Activity}   label="Wave Height" value={liveMetrics["wave_height_meters"]} unit=" m" />}
                {liveMetrics["chlorophyll_mg_m3"]      && <MetricCard Icon={Sprout}     label="Chlorophyll-a" value={liveMetrics["chlorophyll_mg_m3"]} unit=" mg/m³" color="#69f0ae" />}
                {liveMetrics["is_pfz"] !== undefined   && <MetricCard Icon={Fish}       label="Fishing Zone" value={liveMetrics["is_pfz"]} color={String(liveMetrics["is_pfz"]).toLowerCase() === "yes" || liveMetrics["is_pfz"] === "true" ? "#69f0ae" : "#8ab4f8"} />}
              </div>
            </div>
          )}

          <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: 0.5, marginTop: 4, display: "flex", alignItems: "center", gap: 5 }}>
            <Brain size={11} /> Agent Trace
          </div>
          {reasoningTrace.map((t, i) => (
            <div key={i} style={{ background: t.status === "done" ? "rgba(105,240,174,0.05)" : "rgba(0,229,255,0.05)", border: `1px solid ${t.status === "done" ? "rgba(105,240,174,0.18)" : "rgba(0,229,255,0.18)"}`, borderRadius: 8, padding: "8px 12px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                <span style={{ fontSize: "0.65rem", background: t.status === "done" ? "rgba(105,240,174,0.18)" : "rgba(0,229,255,0.18)", color: t.status === "done" ? "#69f0ae" : "var(--accent)", padding: "1px 7px", borderRadius: 10, fontWeight: 700, textTransform: "uppercase", display: "flex", alignItems: "center", gap: 4 }}>
                  {t.status === "done" ? <CheckCircle size={10} /> : <Loader2 size={10} />} {t.agent}
                </span>
              </div>
              <div style={{ fontSize: "0.79rem", color: "var(--text-secondary)", whiteSpace: "pre-wrap", lineHeight: 1.55 }}>{t.detail}</div>
            </div>
          ))}
          {reasoningTrace.length === 0 && <div style={{ color: "gray", fontSize: "0.85rem", textAlign: "center", marginTop: "2rem" }}>Agent activity will appear here after a query.</div>}
        </div>
      </div>
    </div>
  );
}
