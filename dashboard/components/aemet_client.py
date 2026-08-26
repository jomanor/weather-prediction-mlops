import os

import pandas as pd
import requests
import streamlit as st

AEMET_INE_CODES = {
    "Madrid": "28079",
    "Barcelona": "08019",
    "Valencia": "46250",
    "Sevilla": "41091",
    "Zaragoza": "50297",
    "Malaga": "29067",
    "Murcia": "30030",
    "Palma": "07040",
    "Bilbao": "48020",
    "Alicante": "03014",
    "Granada": "18087",
    "Almería": "04013",
    "Paterna": "46190",
    "El Ejido": "04079",
}


@st.cache_data(ttl=3600)
def fetch_aemet_forecast(city: str):
    api_key = os.getenv("AEMET_API_KEY")
    code = AEMET_INE_CODES.get(city)
    if not api_key or not code:
        return pd.DataFrame()

    url = f"https://opendata.aemet.es/openapi/api/prediccion/especifica/municipio/horaria/{code}"
    headers = {"api_key": api_key}

    try:
        res = requests.get(url, headers=headers, timeout=10)
        res_json = res.json()
        if res_json.get("estado") != 200:
            return pd.DataFrame()

        datos_url = res_json.get("datos")
        if not datos_url:
            return pd.DataFrame()

        data_res = requests.get(datos_url, timeout=10)
        data = data_res.json()

        records = []
        for p in data[0]["prediccion"]["dia"]:
            fecha = p.get("fecha")
            for t in p.get("temperatura", []):
                hora = t.get("periodo")
                val = t.get("value")
                if fecha and hora and val:
                    records.append(
                        {"timestamp": f"{fecha[:10]}T{hora}:00:00", "aemet_temp": float(val)}
                    )
        return pd.DataFrame(records)
    except Exception as e:
        st.warning(f"Error fetching AEMET forecast for {city}: {e}")
        return pd.DataFrame()
