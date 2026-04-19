"""WP13 §13.2 — Image captioning pipeline.

Three-tier approach:
  - Small images with text (buttons/labels): local OCR via tesseract (free, instant)
  - Small images without text (icons): Gemini with short icon-ID prompt
  - Large screenshots: Gemini with full description prompt
  - Blank/decorative images: filtered out via entropy + content checks
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from api.core.product import product_labels
from api.core.settings import get_settings

logger = logging.getLogger(__name__)

_REFUSAL_PATTERNS = re.compile(
    r"no puedo|cannot|can't|unable to|borroso|blurry|not relevant|"
    r"no es posible|no relacionad|not related|low quality|too small|"
    r"I'm sorry|lo siento|I cannot",
    re.IGNORECASE,
)

def _captioning_prompt_es() -> str:
    p = product_labels()
    role = p["reader_role_es"]
    sn = p["short_name"]
    return (
        f"Eres un asistente que analiza capturas de pantalla del producto {sn}. "
        "Describe lo que ves con precisión:\n"
        "— Nombre de la pantalla o módulo (si aparece en el encabezado)\n"
        "— Todos los campos del formulario con sus etiquetas y valores visibles\n"
        "— Nombres exactos de los botones y su función aparente\n"
        "— Ruta de navegación visible (Menú > Submenú > Pantalla)\n"
        "— Elementos resaltados, seleccionados o en error\n"
        "— Encabezados de tablas y datos de ejemplo si son visibles\n"
        f"Escribe en español. Sé específico: {role} que lea solo tu descripción "
        "debe poder identificar la pantalla y entender qué acción se muestra.\n\n"
        f"IMPORTANTE: Esta imagen viene de un documento PDF del producto {sn}. "
        "Aunque la calidad sea baja, borrosa o parcial, SIEMPRE describe todo lo que "
        "puedas distinguir. Nunca digas que la imagen no es relevante ni que no puedes "
        "analizarla. Describe cualquier texto, botón, tabla, menú o elemento de interfaz "
        "que puedas identificar, aunque sea parcialmente legible."
    )


def _icon_id_prompt_en() -> str:
    p = product_labels()
    return (
        f"This is a small UI icon or button extracted from a PDF document about {p['long_name_en']} "
        f"({p['domain_en']}). Even if the image quality is low or blurry, identify "
        "what this icon represents in 2-5 words. Examples: 'search icon', 'print icon', "
        "'save button', 'delete icon', 'calendar icon', 'magnifying glass', 'settings gear'. "
        "NEVER say you cannot identify it. Make your best guess. "
        "Reply ONLY with the short description, nothing else."
    )

BUTTON_SIZE_THRESHOLD = (200, 200)
MIN_SCREENSHOT_SIZE = (150, 150)
MIN_ENTROPY = 3.5
MIN_FILESIZE_BYTES = 2048
MIN_UNIQUE_COLORS = 8


def _image_entropy(img: Image.Image) -> float:
    """Shannon entropy of the image histogram — low = blank/solid."""
    histogram = img.histogram()
    total = sum(histogram)
    if total == 0:
        return 0.0
    probs = [h / total for h in histogram if h > 0]
    return -sum(p * math.log2(p) for p in probs)


def _is_thin_strip(img: Image.Image) -> bool:
    """Detect separator lines / progress bars."""
    ratio = max(img.width, img.height) / max(min(img.width, img.height), 1)
    return ratio > 8


def _non_white_ratio(img: Image.Image) -> float:
    """WP13 §13.2 — share of non-near-white pixels (sampled, downscaled)."""
    small = img.resize((80, 80), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())
    if not pixels:
        return 0.0
    non_w = 0
    for px in pixels:
        if len(px) >= 3:
            r, g, b = px[0], px[1], px[2]
        else:
            r = g = b = px[0]
        if not (r > 250 and g > 250 and b > 250):
            non_w += 1
    return non_w / len(pixels)


def classify_image(img_path: Path) -> str:
    """Classify image as 'button', 'screenshot', or 'skip'.

    - button: small inline element with text (icons, buttons, labels)
    - screenshot: larger image with meaningful content
    - skip: blank, decorative, or too low-content to be useful
    """
    try:
        file_size = img_path.stat().st_size
        if file_size < MIN_FILESIZE_BYTES:
            return "skip"

        img = Image.open(img_path)

        if _is_thin_strip(img):
            return "skip"

        if img.mode != "RGB":
            img = img.convert("RGB")

        if img.width < 100 or img.height < 100:
            return "skip"

        if _non_white_ratio(img) < 0.05:
            return "skip"

        entropy = _image_entropy(img)
        if entropy < MIN_ENTROPY:
            return "skip"

        sample = img.resize((min(img.width, 100), min(img.height, 100)))
        unique_colors = len(set(sample.getdata()))
        if unique_colors < MIN_UNIQUE_COLORS:
            return "skip"

        if img.width < BUTTON_SIZE_THRESHOLD[0] or img.height < BUTTON_SIZE_THRESHOLD[1]:
            return "button"

        if img.width < MIN_SCREENSHOT_SIZE[0] or img.height < MIN_SCREENSHOT_SIZE[1]:
            return "skip"

        return "screenshot"
    except Exception as e:
        logger.warning("Image classify error for %s: %s", img_path, e)
        return "skip"


def ocr_button_text(img_path: Path) -> str:
    """Extract text from a small button/icon image using tesseract."""
    try:
        import pytesseract
        img = Image.open(img_path)
        if img.width < 300:
            img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
        text = pytesseract.image_to_string(img, lang="spa+eng", config="--psm 7").strip()
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 2:
            return ""
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio < 0.6:
            return ""
        words = text.split()
        has_real_word = any(
            len(w) >= 4 and w.isalpha() for w in words
        )
        if not has_real_word:
            return ""
        return text
    except Exception as e:
        logger.debug("OCR failed for %s: %s", img_path.name, e)
    return ""


async def identify_icon_gemini(img_path: Path) -> str:
    """Call Gemini to identify a small icon/button image in 2-5 words."""
    settings = get_settings()
    if not settings.google_api_key:
        return ""

    try:
        from google import genai

        from api.core.model_names import gemini_vision_model

        client = genai.Client(api_key=settings.google_api_key)
        model_name = gemini_vision_model()

        img = Image.open(img_path)
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=[_icon_id_prompt_en(), img],
        )
        desc = response.text.strip().rstrip(".")
        if _REFUSAL_PATTERNS.search(desc):
            return "UI element"
        if 1 < len(desc) < 60:
            return desc
    except Exception as e:
        logger.debug("Icon identification failed for %s: %s", img_path.name, e)
    return ""


async def caption_image_gemini(img_path: Path) -> str:
    """Call configured vision model to describe a product UI screenshot."""
    settings = get_settings()
    if not settings.google_api_key:
        logger.warning("GOOGLE_API_KEY not set — returning placeholder caption")
        return f"[Imagen sin descripción: {img_path.name}]"

    try:
        from google import genai

        from api.core.model_names import gemini_vision_model

        client = genai.Client(api_key=settings.google_api_key)
        model_name = gemini_vision_model()

        img = Image.open(img_path)
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=[_captioning_prompt_es(), img],
        )
        caption = response.text.strip()
        if _REFUSAL_PATTERNS.search(caption) and len(caption) < 200:
            logger.warning("Gemini refused caption for %s, using fallback", img_path.name)
            return f"[Captura de pantalla ({product_labels()['short_name']}): {img_path.name}]"
        return caption
    except Exception as e:
        logger.error("Gemini captioning failed for %s: %s", img_path.name, e)
        return f"[Error al describir imagen: {img_path.name}]"


async def caption_image(img_path: Path) -> str:
    """Route to the active vision model."""
    settings = get_settings()
    models_cfg = settings.models_config()
    active = models_cfg.get("vision", {}).get("active", "gemini")

    if active == "gemini":
        return await caption_image_gemini(img_path)
    else:
        return f"[Imagen: {img_path.name} — modelo de visión '{active}' no implementado]"


async def process_images(
    images: list[Path],
    markdown_text: str,
    doc_id: str = "",
) -> tuple[str, int]:
    """Classify, OCR buttons, caption screenshots, embed into markdown."""
    buttons: list[Path] = []
    screenshots: list[Path] = []
    skipped = 0

    for img_path in images:
        cat = classify_image(img_path)
        if cat == "button":
            buttons.append(img_path)
        elif cat == "screenshot":
            screenshots.append(img_path)
        else:
            skipped += 1

    logger.info(
        "Image pipeline: %d extracted → %d buttons, %d screenshots, %d skipped",
        len(images), len(buttons), len(screenshots), skipped,
    )

    captions: list[tuple[str, str]] = []

    for img_path in buttons:
        text = ocr_button_text(img_path)
        if text:
            captions.append((img_path.name, f"(botón: {text})"))
        else:
            icon_desc = await identify_icon_gemini(img_path)
            if icon_desc:
                captions.append((img_path.name, f"(icono: {icon_desc})"))

    for img_path in screenshots:
        cap = await caption_image(img_path)
        captions.append((img_path.name, cap))

    if not captions:
        return markdown_text, 0

    lines = markdown_text.split("\n")
    result_lines: list[str] = []
    page_captions: dict[int, list[tuple[str, str]]] = {}
    for img_name, cap in captions:
        m = re.match(r"image_p(\d+)_", img_name)
        pg = int(m.group(1)) if m else 0
        page_captions.setdefault(pg, []).append((img_name, cap))

    inserted_pages: set[int] = set()
    current_page = 0
    page_heading_re = re.compile(r"^#{1,3}\s+.*(?:Página|Page)\s+(\d+)", re.IGNORECASE)

    for line in lines:
        m = page_heading_re.match(line)
        if m:
            if current_page in page_captions and current_page not in inserted_pages:
                result_lines.extend(_format_captions(page_captions[current_page], doc_id))
                inserted_pages.add(current_page)
            current_page = int(m.group(1))
        result_lines.append(line)

    if current_page in page_captions and current_page not in inserted_pages:
        result_lines.extend(_format_captions(page_captions[current_page], doc_id))
        inserted_pages.add(current_page)

    for pg, caps in page_captions.items():
        if pg not in inserted_pages:
            result_lines.extend(_format_captions(caps, doc_id))

    return "\n".join(result_lines), len(captions)


def _format_captions(caps: list[tuple[str, str]], doc_id: str) -> list[str]:
    lines = [""]
    for img_name, caption in caps:
        img_url = f"/corpus/{doc_id}/media/{img_name}" if doc_id else img_name
        lines.append(f"![{img_name}]({img_url})")
        lines.append(f"> **📷 Imagen:** {caption}")
        lines.append("")
    return lines
