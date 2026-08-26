import os

import pandas as pd
import requests
import streamlit as st


@st.cache_data(ttl=300)
def fetch_supabase_predictions(city: str = None):
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        return pd.DataFrame()

    endpoint = f"{supabase_url}/rest/v1/weather_predictions"
    headers = {"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"}
    params = {"select": "*"}
    if city:
        params["city"] = f"eq.{city}"

    try:
        res = requests.get(endpoint, headers=headers, params=params, timeout=5)
        if res.status_code == 200:
            return pd.DataFrame(res.json())
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
