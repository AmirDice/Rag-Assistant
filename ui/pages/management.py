"""Usage & cost dashboard — model usage, token volume, and spend over time.

Reads /admin/usage/summary (aggregated from the cost meter's event log). Labels
are kept local so we don't touch i18n.py.
"""

import os
from pathlib import Path

import httpx
import streamlit as st
from ui_style import banner, page_heading

API_URL = os.getenv("API_URL", "http://localhost:8000")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOTENV_PATH = _REPO_ROOT / ".env"


def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return out
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, value = s.split("=", 1)
        k = key.strip()
        if not k:
            continue
        v = value.strip()
        if " #" in v:
            v = v.split(" #", 1)[0].strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        out[k] = v
    return out


def _admin_headers() -> dict[str, str]:
    env_file = _read_env_file(_DOTENV_PATH)
    token = (os.getenv("ADMIN_TOKEN", "") or env_file.get("ADMIN_TOKEN", "")).strip()
    return {"X-Admin-Token": token} if token else {}


_L = {
    "es": {
        "title": "Uso y costes",
        "desc": "Consumo de modelos de generación: número de respuestas, tokens y coste estimado a lo largo del tiempo (a partir del registro del medidor de costes).",
        "empty": "Aún no hay datos de uso. Genera algunas respuestas en el Chat (con la IA activada) y vuelve aquí.",
        "total_cost": "Coste total",
        "generations": "Respuestas generadas",
        "tokens_in": "Tokens de entrada",
        "tokens_out": "Tokens de salida",
        "cost_by_model": "Coste por modelo",
        "cost_over_time": "Coste por día",
        "gens_by_model": "Respuestas por modelo",
        "by_model_table": "Detalle por modelo",
        "recent": "Eventos recientes",
        "load_error": "No se pudieron cargar los datos de uso",
        "col_model": "Modelo",
        "col_provider": "Proveedor",
        "col_cost": "Coste",
        "col_events": "Respuestas",
        "col_in": "Tokens entrada",
        "col_out": "Tokens salida",
    },
    "en": {
        "title": "Usage & cost",
        "desc": "Generation-model consumption: number of answers, tokens, and estimated cost over time (from the cost meter's event log).",
        "empty": "No usage yet. Generate a few answers in Chat (AI toggle on) and come back.",
        "total_cost": "Total cost",
        "generations": "Answers generated",
        "tokens_in": "Input tokens",
        "tokens_out": "Output tokens",
        "cost_by_model": "Cost by model",
        "cost_over_time": "Cost per day",
        "gens_by_model": "Answers by model",
        "by_model_table": "Per-model detail",
        "recent": "Recent events",
        "load_error": "Could not load usage data",
        "col_model": "Model",
        "col_provider": "Provider",
        "col_cost": "Cost",
        "col_events": "Answers",
        "col_in": "Input tokens",
        "col_out": "Output tokens",
    },
}


def _l(key: str) -> str:
    lang = st.session_state.get("ui_lang", "en")
    return _L.get(lang, _L["en"]).get(key, key)


page_heading(_l("title"), "monitoring")
st.caption(_l("desc"))


def _fmt_cost(v: float, cur: str) -> str:
    return f"${v:,.4f}" if cur == "USD" else f"{v:,.4f} {cur}"


def _management_body() -> None:
    try:
        r = httpx.get(f"{API_URL}/admin/usage/summary", headers=_admin_headers(), timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        banner(f"{_l('load_error')}: {e}", variant="error", icon_name="error")
        return

    currency = data.get("currency", "USD")
    total_cost = float(data.get("total_cost") or 0.0)
    total_events = int(data.get("total_events") or 0)
    by_model = data.get("by_model") or []
    by_day = data.get("by_day") or []
    recent = data.get("recent") or []

    if total_events == 0:
        banner(_l("empty"), variant="info", icon_name="insights")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(_l("total_cost"), _fmt_cost(total_cost, currency))
    c2.metric(_l("generations"), f"{total_events:,}")
    c3.metric(_l("tokens_in"), f"{int(data.get('total_units_in') or 0):,}")
    c4.metric(_l("tokens_out"), f"{int(data.get('total_units_out') or 0):,}")

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader(_l("cost_by_model"))
        st.bar_chart(
            [{"model": m["model"], "cost": round(float(m["cost"]), 6)} for m in by_model],
            x="model", y="cost", color="#6c5ce7", height=280,
        )
    with right:
        st.subheader(_l("gens_by_model"))
        st.bar_chart(
            [{"model": m["model"], "answers": int(m["events"])} for m in by_model],
            x="model", y="answers", color="#00b894", height=280,
        )

    st.subheader(_l("cost_over_time"))
    st.area_chart(
        [{"date": d["date"], "cost": round(float(d["cost"]), 6)} for d in by_day],
        x="date", y="cost", color="#6c5ce7", height=260,
    )

    st.subheader(_l("by_model_table"))
    st.dataframe(
        [
            {
                _l("col_model"): m["model"],
                _l("col_provider"): m.get("provider", ""),
                _l("col_cost"): round(float(m["cost"]), 6),
                _l("col_events"): int(m["events"]),
                _l("col_in"): int(m["units_in"]),
                _l("col_out"): int(m["units_out"]),
            }
            for m in by_model
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(_l("recent")):
        st.dataframe(
            [
                {
                    "ts": e.get("ts", ""),
                    "model": e.get("model", ""),
                    "provider": e.get("provider", ""),
                    "cost": round(float(e.get("estimated_cost") or 0.0), 6),
                    "in": int(e.get("units_in") or 0),
                    "out": int(e.get("units_out") or 0),
                }
                for e in recent
            ],
            use_container_width=True,
            hide_index=True,
        )


_management_body()
