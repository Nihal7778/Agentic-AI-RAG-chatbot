"""
Open-Meteo Weather Tool for Agentic RAG Chatbot (Feature C).
Safe external API call + sandboxed analytics.

"""

import re
import json
import requests
import numpy as np
from typing import Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import LLM_MODEL, LLM_TEMPERATURE

BASE_URL = "https://api.open-meteo.com/v1/forecast"
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"

WEATHER_KEYWORDS = [
    "weather", "temperature", "forecast", "climate",
    "rain", "wind", "humidity", "snow", "storm",
    "celsius", "fahrenheit", "open-meteo", "open meteo"
]

LOCATION_PATTERNS = [
    r"weather (?:in|for|at) (.+?)(?:\?|$|\.)",
    r"temperature (?:in|for|at) (.+?)(?:\?|$|\.)",
    r"forecast (?:in|for|at) (.+?)(?:\?|$|\.)",
    r"how (?:hot|cold|warm) is (?:it in )?(.+?)(?:\?|$|\.)",
    r"(?:weather|climate) (?:data|info|analysis) (?:for|in|of) (.+?)(?:\?|$|\.)",
    r"(?:show|get|fetch) (?:me )?weather (?:for|in) (.+?)(?:\?|$|\.)",
]

WEATHER_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["location", "analytics"],
    template="""You are a weather data analyst. Given analytics computed from
Open-Meteo time series data, provide a clear summary.

Location: {location}
Analytics: {analytics}

Provide a concise summary covering:
1. Temperature overview and trend
2. Wind conditions
3. Any anomalies detected
4. Data quality
Keep it to 4-5 sentences. Be specific with numbers."""
)


class WeatherTool:
    """Open-Meteo weather analysis tool with sandboxed computation."""

    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, max_tokens=500)
        self.chain = WEATHER_SUMMARY_PROMPT | self.llm | StrOutputParser()

    @staticmethod
    def detect_weather_query(query: str) -> Optional[str]:
        """
        Check if query is weather-related AND has a location.
        Returns location string or None.
        Requires BOTH keyword + location to avoid false positives.
        """
        q = query.lower().strip()

        if not any(kw in q for kw in WEATHER_KEYWORDS):
            return None

        for pattern in LOCATION_PATTERNS:
            m = re.search(pattern, q, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip("?.!")

        return None  # Has keyword but no location → don't trigger

    @staticmethod
    def geocode(city: str) -> Optional[Dict]:
        """Convert city name to lat/lon."""
        try:
            res = requests.get(GEO_URL, params={"name": city, "count": 1}, timeout=5)
            data = res.json()
            if "results" in data and data["results"]:
                r = data["results"][0]
                return {
                    "name": r["name"],
                    "country": r.get("country", ""),
                    "latitude": r["latitude"],
                    "longitude": r["longitude"],
                    "timezone": r.get("timezone", "UTC")
                }
        except Exception as e:
            print(f"⚠️ Geocoding failed: {e}")
        return None

    @staticmethod
    def fetch_weather(lat: float, lon: float, tz: str = "UTC") -> Optional[Dict]:
        """Fetch weather data from Open-Meteo API."""
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,wind_speed_10m,relative_humidity_2m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": tz,
                "past_days": 7,
                "forecast_days": 7
            }
            res = requests.get(BASE_URL, params=params, timeout=10)
            return res.json()
        except Exception as e:
            print(f"⚠️ Weather fetch failed: {e}")
        return None

    @staticmethod
    def compute_analytics(data: Dict) -> Dict:
        """
        Sandboxed analytics computation.
        Runs numpy operations on weather time series.
        No file access, no side effects — pure computation.
        """
        hourly = data["hourly"]
        temps = np.array([t if t is not None else np.nan for t in hourly["temperature_2m"]])
        winds = np.array([w if w is not None else np.nan for w in hourly["wind_speed_10m"]])
        humid = np.array([h if h is not None else np.nan for h in hourly["relative_humidity_2m"]])
        times = hourly["time"]

        analytics = {}

        # Temperature
        analytics["temperature"] = {
            "mean": round(float(np.nanmean(temps)), 2),
            "std": round(float(np.nanstd(temps)), 2),
            "min": round(float(np.nanmin(temps)), 2),
            "max": round(float(np.nanmax(temps)), 2),
        }

        # Rolling average (24h)
        window = 24
        if len(temps) >= window:
            rolling = np.convolve(np.nan_to_num(temps), np.ones(window)/window, mode='valid')
            analytics["temperature"]["trend"] = "rising" if rolling[-1] > rolling[0] else "falling"

        # Wind
        analytics["wind"] = {
            "mean": round(float(np.nanmean(winds)), 2),
            "max": round(float(np.nanmax(winds)), 2),
            "calm_hours": int(np.sum(winds < 5)),
            "gusty_hours": int(np.sum(winds > 30)),
        }

        # Humidity
        analytics["humidity"] = {
            "mean": round(float(np.nanmean(humid)), 2),
            "min": round(float(np.nanmin(humid)), 2),
            "max": round(float(np.nanmax(humid)), 2),
        }

        # Data quality
        total = len(temps)
        analytics["data_quality"] = {
            "total_points": total,
            "missing_temp": int(np.sum(np.isnan(temps))),
            "missing_wind": int(np.sum(np.isnan(winds))),
            "completeness_pct": round((1 - np.mean(np.isnan(temps))) * 100, 1)
        }

        # Anomaly detection (beyond 2 std devs)
        t_mean, t_std = np.nanmean(temps), np.nanstd(temps)
        anomalies = np.where(np.abs(temps - t_mean) > 2 * t_std)[0]
        analytics["anomalies"] = {
            "count": int(len(anomalies)),
            "threshold": f"±{round(2*t_std, 1)}°C from mean",
            "timestamps": [times[i] for i in anomalies[:3]]
        }

        # Volatility
        diffs = np.abs(np.diff(np.nan_to_num(temps)))
        analytics["volatility"] = {
            "mean_hourly_change": round(float(np.mean(diffs)), 2),
            "max_hourly_change": round(float(np.max(diffs)), 2),
        }

        return analytics

    def analyze(self, city: str) -> Dict:
        """
        Full pipeline: geocode → fetch → analyze → summarize.
        Returns structured result for the orchestrator.
        """
        # Geocode
        geo = self.geocode(city)
        if not geo:
            return {
                "answer": f"Sorry, I couldn't find the location '{city}'. Try a major city name.",
                "tool": "open_meteo",
                "success": False
            }

        # Fetch
        data = self.fetch_weather(geo["latitude"], geo["longitude"], geo["timezone"])
        if not data or "hourly" not in data:
            return {
                "answer": f"Failed to fetch weather data for {geo['name']}. The API might be temporarily unavailable.",
                "tool": "open_meteo",
                "success": False
            }

        # Compute analytics (sandboxed)
        analytics = self.compute_analytics(data)

        # LLM summary
        try:
            summary = self.chain.invoke({
                "location": f"{geo['name']}, {geo['country']}",
                "analytics": json.dumps(analytics, indent=2)
            })
        except Exception as e:
            summary = f"Analytics computed for {geo['name']}: Temp {analytics['temperature']['min']}–{analytics['temperature']['max']}°C, trend {analytics['temperature'].get('trend', 'stable')}."

        return {
            "answer": f"🌤️ **Weather Analysis: {geo['name']}, {geo['country']}**\n\n{summary}",
            "tool": "open_meteo",
            "location": geo,
            "analytics": analytics,
            "success": True
        }