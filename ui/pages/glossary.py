"""Glossary admin — edit the query-preprocessing terms (CRUD over /admin/glossary).

Lets you fix terminology mappings (canonical term + aliases/typos + a
recall-widening expansion) without a deploy. Labels are kept local here so we
don't touch the (user-modified) i18n.py.
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
        "title": "Glosario",
        "desc": "Edita los términos que enriquecen las consultas antes de la búsqueda: forma canónica, alias/erratas que se detectan, y expansión para ampliar la recuperación.",
        "tenant_filter": "Filtrar por tenant (vacío = global + todos)",
        "empty": "No hay entradas todavía.",
        "col_aliases": "Alias",
        "col_expansion": "Expansión",
        "col_enabled": "Activo",
        "add": "Añadir entrada",
        "term": "Término canónico",
        "aliases_help": "Alias / erratas (separados por comas)",
        "expansion": "Expansión (tokens extra para recuperación)",
        "description": "Descripción (solo para administración)",
        "tenant_opt": "Tenant (vacío = global)",
        "enabled": "Activo",
        "create": "Crear",
        "term_required": "El término es obligatorio.",
        "created": "Entrada creada.",
        "edit_delete": "Editar / eliminar por id",
        "entry_id": "ID de la entrada",
        "set_enabled": "Aplicar activo/inactivo",
        "updated": "Entrada actualizada.",
        "delete": "Eliminar",
        "deleted": "Entrada eliminada.",
        "load_error": "No se pudo cargar el glosario",
        "save_error": "No se pudo guardar",
        "delete_error": "No se pudo eliminar",
    },
    "en": {
        "title": "Glossary",
        "desc": "Edit the terms that enrich queries before retrieval: canonical form, aliases/typos that get detected, and an expansion to widen recall.",
        "tenant_filter": "Filter by tenant (empty = global + all)",
        "empty": "No entries yet.",
        "col_aliases": "Aliases",
        "col_expansion": "Expansion",
        "col_enabled": "Enabled",
        "add": "Add entry",
        "term": "Canonical term",
        "aliases_help": "Aliases / typos (comma-separated)",
        "expansion": "Expansion (extra retrieval tokens)",
        "description": "Description (admin only)",
        "tenant_opt": "Tenant (empty = global)",
        "enabled": "Enabled",
        "create": "Create",
        "term_required": "Term is required.",
        "created": "Entry created.",
        "edit_delete": "Edit / delete by id",
        "entry_id": "Entry id",
        "set_enabled": "Apply enabled/disabled",
        "updated": "Entry updated.",
        "delete": "Delete",
        "deleted": "Entry deleted.",
        "load_error": "Could not load glossary",
        "save_error": "Could not save",
        "delete_error": "Could not delete",
    },
}


def _l(key: str) -> str:
    lang = st.session_state.get("ui_lang", "en")
    return _L.get(lang, _L["en"]).get(key, key)


page_heading(_l("title"), "menu_book")
st.caption(_l("desc"))


def _glossary_body() -> None:
    headers = _admin_headers()
    tenant = st.text_input(_l("tenant_filter"), value="")
    params = {"tenant_id": tenant.strip()} if tenant.strip() else {}

    try:
        r = httpx.get(f"{API_URL}/admin/glossary", params=params, headers=headers, timeout=10)
        r.raise_for_status()
        entries = r.json().get("entries", [])
    except Exception as e:
        banner(f"{_l('load_error')}: {e}", variant="error", icon_name="error")
        entries = []

    if entries:
        st.dataframe(
            [
                {
                    "id": e.get("id"),
                    _l("term"): e.get("term", ""),
                    _l("col_aliases"): ", ".join(e.get("aliases") or []),
                    _l("col_expansion"): e.get("expansion", ""),
                    "tenant": e.get("tenant_id", ""),
                    _l("col_enabled"): bool(e.get("enabled", True)),
                }
                for e in entries
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption(_l("empty"))

    with st.expander(_l("add")):
        with st.form("glossary_add", clear_on_submit=True):
            term = st.text_input(_l("term"))
            aliases = st.text_input(_l("aliases_help"))
            expansion = st.text_input(_l("expansion"))
            description = st.text_area(_l("description"))
            tenant_id = st.text_input(_l("tenant_opt"))
            enabled = st.checkbox(_l("enabled"), value=True)
            if st.form_submit_button(_l("create")):
                if not term.strip():
                    banner(_l("term_required"), variant="warn", icon_name="warning")
                else:
                    payload = {
                        "term": term.strip(),
                        "aliases": [a.strip() for a in aliases.split(",") if a.strip()],
                        "expansion": expansion.strip(),
                        "description": description.strip(),
                        "tenant_id": tenant_id.strip(),
                        "enabled": bool(enabled),
                    }
                    try:
                        resp = httpx.post(
                            f"{API_URL}/admin/glossary", json=payload, headers=headers, timeout=10
                        )
                        resp.raise_for_status()
                        banner(_l("created"), variant="ok", icon_name="check_circle")
                        st.rerun()
                    except Exception as e:
                        banner(f"{_l('save_error')}: {e}", variant="error", icon_name="error")

    with st.expander(_l("edit_delete")):
        eid = st.number_input(_l("entry_id"), min_value=1, step=1, value=1)
        new_enabled = st.checkbox(_l("enabled"), value=True, key="glossary_edit_enabled")
        c1, c2 = st.columns(2)
        with c1:
            if st.button(_l("set_enabled"), use_container_width=True):
                try:
                    resp = httpx.patch(
                        f"{API_URL}/admin/glossary/{int(eid)}",
                        json={"enabled": bool(new_enabled)},
                        headers=headers,
                        timeout=10,
                    )
                    resp.raise_for_status()
                    banner(_l("updated"), variant="ok", icon_name="check_circle")
                    st.rerun()
                except Exception as e:
                    banner(f"{_l('save_error')}: {e}", variant="error", icon_name="error")
        with c2:
            if st.button(_l("delete"), use_container_width=True):
                try:
                    resp = httpx.delete(
                        f"{API_URL}/admin/glossary/{int(eid)}", headers=headers, timeout=10
                    )
                    resp.raise_for_status()
                    banner(_l("deleted"), variant="ok", icon_name="check_circle")
                    st.rerun()
                except Exception as e:
                    banner(f"{_l('delete_error')}: {e}", variant="error", icon_name="error")


_glossary_body()
