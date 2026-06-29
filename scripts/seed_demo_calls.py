"""Seed synthetic, analyzed support calls into the calls catalog (no ASR needed).

Writes a few generic 'Acme Cloud support' CallAnalysis records into the calls
catalog SQLite + the audio-pipeline output JSON, so the Calls page has data to
show (list + detail) for the demo. Run once:

    python scripts/seed_demo_calls.py

Reset later by deleting data/audio_calls_catalog.sqlite and the seeded output dirs.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.core.settings import get_settings
from modules.audio_pipeline.calls_catalog import open_catalog_db, upsert_call
from modules.audio_pipeline.schemas import CallAnalysis, RAGPair, TranscriptLine

AGENT_ID = "call_audio"
ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = Path(get_settings().data_dir).resolve() / "audio_calls_catalog.sqlite"
OUTPUT_ROOT = ROOT / "modules" / "audio_pipeline" / "output"


def _hash(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _line(start: float, end: float, speaker: str, text: str) -> TranscriptLine:
    return TranscriptLine(start=start, end=end, speaker=speaker, text=text)


CALLS = [
    CallAnalysis(
        call_id="CALL-001",
        source_file="acme_support_sso_setup.mp3",
        source_file_hash=_hash("acme_support_sso_setup"),
        timestamp_start="00:00",
        timestamp_end="05:12",
        farmacia="Northwind Traders",
        llamante="Jordan Lee",
        agent="Sam (Acme Support)",
        problema_corto="SSO sign-in fails for all members after enabling SAML.",
        descripcion_problema=(
            "The customer enabled SAML single sign-on but members are redirected back with an "
            "error and cannot sign in. The identity provider metadata URL was entered but the "
            "email attribute mapping was left blank."
        ),
        causa_raiz="Email attribute was not mapped in the SSO configuration.",
        resolucion=(
            "1. Open Admin Settings > SSO. 2. Confirm the identity provider metadata URL. "
            "3. Map the email and name attributes. 4. Assign a default role for new users. "
            "5. Save and retry sign-in."
        ),
        resolucion_exitosa=True,
        resumen=(
            "Customer's SAML SSO failed because the email attribute was unmapped. Agent walked "
            "them through mapping email/name attributes in Admin Settings > SSO; sign-in worked."
        ),
        rag_qa=[
            RAGPair(
                question="Why do members get an error after I enable SSO?",
                answer=(
                    "Map the email and name attributes in Admin Settings > SSO and set a default "
                    "role for new users, then retry. An unmapped email attribute blocks sign-in."
                ),
                category="authentication",
                confidence=0.9,
            ),
        ],
        software_features=["SSO", "SAML", "Admin Settings"],
        tags=["sso", "authentication", "resolved"],
        transcript=[
            _line(0.0, 6.0, "caller", "Hi, I turned on single sign-on and now nobody on my team can log in."),
            _line(6.0, 12.0, "agent", "Let's check the SSO config. Did you map the email attribute under Admin Settings, SSO?"),
            _line(12.0, 18.0, "caller", "I added the metadata URL but I don't think I mapped any attributes."),
            _line(18.0, 26.0, "agent", "That's the cause. Map email and name, set a default role, and save. Then try signing in."),
            _line(26.0, 30.0, "caller", "That worked, thank you!"),
        ],
    ),
    CallAnalysis(
        call_id="CALL-002",
        source_file="acme_support_upload_fail.mp3",
        source_file_hash=_hash("acme_support_upload_fail"),
        timestamp_start="00:00",
        timestamp_end="03:48",
        farmacia="Globex Inc.",
        llamante="Priya Nair",
        agent="Sam (Acme Support)",
        problema_corto="Large file uploads fail in the browser.",
        descripcion_problema=(
            "Uploads of large files fail partway through in the browser. The workspace has plenty "
            "of remaining storage and the file is under the plan's per-file limit."
        ),
        causa_raiz="Browser extension was interrupting the upload; large files need the desktop sync app.",
        resolucion=(
            "1. Confirm the file is under the per-file size limit. 2. Disable upload-blocking browser "
            "extensions. 3. For large files, use the desktop sync app instead of the browser."
        ),
        resolucion_exitosa=True,
        resumen=(
            "Large browser uploads failed due to a browser extension. Agent advised disabling the "
            "extension and using the desktop sync app for large files, which resolved it."
        ),
        rag_qa=[
            RAGPair(
                question="My large file upload keeps failing — what should I do?",
                answer=(
                    "Check the file is under the per-file size limit, disable upload-blocking browser "
                    "extensions, and use the desktop sync app for large files."
                ),
                category="uploads",
                confidence=0.88,
            ),
        ],
        software_features=["Uploads", "Desktop sync"],
        tags=["uploads", "sync", "resolved"],
        transcript=[
            _line(0.0, 7.0, "caller", "Every time I upload a big file it fails about halfway."),
            _line(7.0, 14.0, "agent", "Is the file under your plan's per-file limit, and do you have storage left?"),
            _line(14.0, 19.0, "caller", "Yes to both."),
            _line(19.0, 27.0, "agent", "Try disabling browser extensions, and for large files use the desktop sync app."),
            _line(27.0, 31.0, "caller", "The sync app did it. Appreciate the help."),
        ],
    ),
    CallAnalysis(
        call_id="CALL-003",
        source_file="acme_support_billing_plan.mp3",
        source_file_hash=_hash("acme_support_billing_plan"),
        timestamp_start="00:00",
        timestamp_end="02:55",
        farmacia="Initech",
        llamante="Marcus Cole",
        agent="Dana (Acme Support)",
        problema_corto="Wants SSO but it's missing on the Team plan.",
        descripcion_problema=(
            "Customer cannot find the SSO option. They are on the Team plan; SSO is a Business-plan "
            "feature."
        ),
        causa_raiz="SSO is only available on the Business plan.",
        resolucion="Explained plan differences; customer will upgrade to Business to get SSO.",
        resolucion_exitosa=False,
        resumen=(
            "Customer on the Team plan couldn't find SSO. Agent explained SSO requires the Business "
            "plan; customer will consider upgrading. Not resolved in-call."
        ),
        rag_qa=[
            RAGPair(
                question="Which plan do I need for single sign-on?",
                answer="Single sign-on (SAML/OIDC) is available on the Business plan.",
                category="billing",
                confidence=0.95,
            ),
        ],
        software_features=["Billing", "Plans", "SSO"],
        tags=["billing", "plans", "unresolved"],
        transcript=[
            _line(0.0, 6.0, "caller", "I can't find the single sign-on setting anywhere."),
            _line(6.0, 12.0, "agent", "Which plan are you on? SSO is a Business-plan feature."),
            _line(12.0, 16.0, "caller", "We're on Team. So I'd need to upgrade?"),
            _line(16.0, 22.0, "agent", "Correct — upgrading to Business unlocks SSO and advanced audit logs."),
        ],
    ),
]


def main() -> None:
    conn = open_catalog_db(CATALOG_PATH)
    try:
        for ca in CALLS:
            upsert_call(conn, agent_id=AGENT_ID, ca=ca)
            out_dir = OUTPUT_ROOT / ca.source_file_hash
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{ca.call_id}.json").write_text(
                ca.model_dump_json(indent=2), encoding="utf-8"
            )
            print(f"seeded {ca.call_id}: {ca.problema_corto}")
        conn.commit()
    finally:
        conn.close()
    print(f"done — {len(CALLS)} calls under agent '{AGENT_ID}' (catalog: {CATALOG_PATH})")


if __name__ == "__main__":
    main()
