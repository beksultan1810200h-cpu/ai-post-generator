import base64
import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone

import requests
from flask import Flask, request, jsonify, g, send_from_directory, make_response
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")

# Hugging Face Inference Providers API (chat / text). Get a FREE token at
# https://huggingface.co/settings/tokens and set it as the HF_API_KEY
# environment variable on your hosting platform.
# NOTE: the old "api-inference.huggingface.co/models/<id>" endpoint was
# permanently discontinued by Hugging Face (returns 410 / fails DNS lookup).
# All text inference now goes through the OpenAI-compatible chat-completions
# router below.
HF_API_KEY = os.environ.get("HF_API_KEY", "")
HF_MODEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

# Hugging Face image generation (text-to-image) goes through the
# provider-agnostic "hf-inference" REST route rather than the chat
# completions route (image generation is not part of the OpenAI-compatible
# chat API). This is used as a FREE fallback when REPLICATE_API_TOKEN is
# not configured.
HF_IMAGE_MODEL = os.environ.get("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-2-1")
HF_IMAGE_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}"

# Replicate (optional, paid after free credits run out). Sign up at
# https://replicate.com, grab a token at https://replicate.com/account/api-tokens
# and set REPLICATE_API_TOKEN. New accounts get some free trial credit.
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
REPLICATE_MODEL_VERSION = os.environ.get(
    # black-forest-labs/flux-schnell is fast & cheap; override via env if needed
    # (e.g. "stability-ai/stable-diffusion-3").
    "REPLICATE_MODEL",
    "black-forest-labs/flux-schnell",
)
REPLICATE_PREDICTIONS_URL = "https://api.replicate.com/v1/models/{model}/predictions"

SESSION_COOKIE_NAME = "card_session_id"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # 1 year

PHOTO_STYLES = {
    "studio": "professional studio product photography, clean seamless white background, softbox lighting, high detail",
    "model": "lifestyle photo of a person using the product naturally, realistic, soft daylight, editorial style",
    "interior": "product placed in a cozy modern interior scene, natural context, warm ambient lighting",
    "closeup": "extreme macro close-up of the product surface and texture, sharp focus, studio lighting",
}
PHOTO_STYLE_LABELS = {
    "studio": "Студия",
    "model": "Модель",
    "interior": "Интерьер",
    "closeup": "Крупный план",
}

CARD_STYLE_PRESETS = {"minimal", "luxury", "kids", "techno", "eco"}

app = Flask(__name__, static_folder=None)
CORS(app, supports_credentials=True)  # allow requests from any origin


# --------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    # New table for the product-card constructor. full_data stores the
    # entire card state (blocks, style, texts, photo urls) as a JSON string
    # so the editor can be fully restored later.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            title TEXT NOT NULL,
            full_data TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


# --------------------------------------------------------------------------
# Session (cookie) helpers for the card constructor
# --------------------------------------------------------------------------
def get_session_id():
    """Returns (session_id, is_new). Does not set the cookie itself -
    call attach_session_cookie() on the outgoing response if is_new."""
    sid = request.cookies.get(SESSION_COOKIE_NAME)
    if sid:
        return sid, False
    return str(uuid.uuid4()), True


def attach_session_cookie(resp, session_id, is_new):
    if is_new:
        resp.set_cookie(
            SESSION_COOKIE_NAME,
            session_id,
            max_age=SESSION_COOKIE_MAX_AGE,
            httponly=True,
            samesite="Lax",
        )
    return resp


# --------------------------------------------------------------------------
# Hugging Face: text generation
# --------------------------------------------------------------------------
def call_huggingface_chat(system_prompt: str, user_prompt: str, max_tokens=500, temperature=0.7) -> str:
    if not HF_API_KEY:
        raise RuntimeError(
            "HF_API_KEY не задан. Получите бесплатный токен на "
            "https://huggingface.co/settings/tokens и добавьте его как "
            "переменную окружения HF_API_KEY."
        )

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=90)

    if resp.status_code != 200:
        raise RuntimeError(f"HuggingFace API error {resp.status_code}: {resp.text[:300]}")

    data = resp.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"HuggingFace API error: {data['error']}")
        raise RuntimeError(f"Неожиданный формат ответа HuggingFace: {str(data)[:300]}")


def call_huggingface(prompt: str) -> str:
    """Kept for backward compatibility with the original /generate route."""
    return call_huggingface_chat(
        "Ты — маркетолог-копирайтер. Пиши короткие продающие посты на русском.",
        f"Напиши короткий продающий пост на основе описания товара: {prompt}",
    )


def extract_json(text: str):
    """The model is asked to reply with pure JSON, but instruct-models
    sometimes wrap it in ```json fences or add stray text. This pulls out
    the first {...} block and parses it."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Модель не вернула корректный JSON")


def generate_card_content(title, description, advantages_raw):
    system_prompt = (
        "Ты — профессиональный копирайтер маркетплейсов. Отвечай ТОЛЬКО валидным JSON, "
        "без markdown-разметки, без пояснений до или после. Пиши на русском языке."
    )
    user_prompt = f"""
Составь контент для карточки товара на маркетплейсе.
Название товара: {title}
Описание от продавца: {description}
Ключевые преимущества (черновик от продавца): {advantages_raw or "не указаны, придумай сам"}

Верни JSON строго со следующей структурой (без лишних полей):
{{
  "headline": "цепляющий заголовок карточки, до 70 символов",
  "descriptions": ["вариант описания 1 (2-3 предложения)", "вариант описания 2", "вариант описания 3"],
  "advantages": ["преимущество 1", "преимущество 2", "преимущество 3", "преимущество 4", "преимущество 5"],
  "characteristics": [
    {{"name": "Характеристика 1", "value": "значение"}},
    {{"name": "Характеристика 2", "value": "значение"}},
    {{"name": "Характеристика 3", "value": "значение"}},
    {{"name": "Характеристика 4", "value": "значение"}},
    {{"name": "Характеристика 5", "value": "значение"}},
    {{"name": "Характеристика 6", "value": "значение"}}
  ]
}}
""".strip()

    raw = call_huggingface_chat(system_prompt, user_prompt, max_tokens=900, temperature=0.6)
    data = extract_json(raw)

    data.setdefault("headline", title)
    data.setdefault("descriptions", [description] if description else [""])
    data.setdefault("advantages", [])
    data.setdefault("characteristics", [])
    return data


# --------------------------------------------------------------------------
# Image generation: Replicate (preferred) with Hugging Face fallback
# --------------------------------------------------------------------------
def generate_image_replicate(prompt: str) -> str:
    """Returns a data URI (base64) or a direct image URL. Raises on failure."""
    headers = {
        "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait",  # ask Replicate to block until the prediction finishes (up to ~60s)
    }
    url = REPLICATE_PREDICTIONS_URL.format(model=REPLICATE_MODEL_VERSION)
    payload = {"input": {"prompt": prompt}}

    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Replicate error {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    output = data.get("output")

    # Replicate may still be processing even with Prefer: wait on cold starts;
    # poll the prediction URL for a short while if so.
    get_url = data.get("urls", {}).get("get")
    attempts = 0
    while data.get("status") in ("starting", "processing") and get_url and attempts < 20:
        import time

        time.sleep(2)
        poll = requests.get(get_url, headers=headers, timeout=30)
        data = poll.json()
        output = data.get("output")
        attempts += 1

    if data.get("status") == "failed":
        raise RuntimeError(f"Replicate prediction failed: {data.get('error')}")

    if isinstance(output, list) and output:
        return output[0]
    if isinstance(output, str) and output:
        return output

    raise RuntimeError("Replicate не вернул изображение")


def generate_image_hf(prompt: str) -> str:
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY не задан, fallback-генерация изображений недоступна.")

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(HF_IMAGE_API_URL, headers=headers, json={"inputs": prompt}, timeout=120)

    if resp.status_code != 200:
        raise RuntimeError(f"HuggingFace image API error {resp.status_code}: {resp.text[:300]}")

    content_type = resp.headers.get("content-type", "")
    if content_type.startswith("image/"):
        b64 = base64.b64encode(resp.content).decode("utf-8")
        return f"data:{content_type};base64,{b64}"

    # Some providers return JSON with a base64 field instead of raw bytes
    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError("Неожиданный ответ от HuggingFace image API")

    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"HuggingFace image API error: {data['error']}")

    raise RuntimeError(f"Неожиданный формат ответа HuggingFace image API: {str(data)[:300]}")


def generate_image(prompt: str) -> str:
    if REPLICATE_API_TOKEN:
        try:
            return generate_image_replicate(prompt)
        except Exception as e:  # fall through to HF on any Replicate failure
            print(f"[generate_image] Replicate failed, falling back to HF: {e}")

    return generate_image_hf(prompt)


# --------------------------------------------------------------------------
# Routes: original post generator (UNCHANGED)
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/ping")
def ping():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    user_id = (data.get("user_id") or "").strip() or str(uuid.uuid4())

    if not prompt:
        return jsonify({"error": "Поле 'prompt' обязательно"}), 400

    try:
        result = call_huggingface(prompt)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Сетевая ошибка при обращении к HuggingFace: {e}"}), 502

    created_at = datetime.now(timezone.utc).isoformat()

    db = get_db()
    db.execute(
        "INSERT INTO requests (user_id, prompt, result, created_at) VALUES (?, ?, ?, ?)",
        (user_id, prompt, result, created_at),
    )
    db.commit()

    return jsonify(
        {
            "user_id": user_id,
            "prompt": prompt,
            "result": result,
            "created_at": created_at,
        }
    )


@app.route("/history", methods=["GET"])
def history():
    user_id = request.args.get("user_id", "").strip()
    db = get_db()

    if user_id:
        rows = db.execute(
            "SELECT id, user_id, prompt, result, created_at FROM requests "
            "WHERE user_id = ? ORDER BY id DESC LIMIT 100",
            (user_id,),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT id, user_id, prompt, result, created_at FROM requests "
            "ORDER BY id DESC LIMIT 100"
        ).fetchall()

    return jsonify([dict(row) for row in rows])


@app.route("/stats", methods=["GET"])
def stats():
    db = get_db()
    total = db.execute("SELECT COUNT(*) AS c FROM requests").fetchone()["c"]
    users = db.execute("SELECT COUNT(DISTINCT user_id) AS c FROM requests").fetchone()["c"]
    return jsonify({"total_requests": total, "unique_users": users})


# --------------------------------------------------------------------------
# Routes: product card constructor (NEW)
# --------------------------------------------------------------------------
@app.route("/generate_card", methods=["POST"])
def generate_card():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()
    advantages_raw = (data.get("advantages") or "").strip()
    style = (data.get("style") or "minimal").strip()
    photo_type = (data.get("photo_type") or "studio").strip()

    if not title:
        return jsonify({"error": "Поле 'title' обязательно"}), 400
    if style not in CARD_STYLE_PRESETS:
        style = "minimal"
    if photo_type not in PHOTO_STYLES:
        photo_type = "studio"

    session_id, is_new = get_session_id()

    try:
        content = generate_card_content(title, description, advantages_raw)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except ValueError as e:
        return jsonify({"error": f"Не удалось разобрать ответ модели: {e}"}), 502
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Сетевая ошибка при обращении к HuggingFace: {e}"}), 502

    payload = {
        "title": title,
        "style": style,
        "photo_type": photo_type,
        "headline": content.get("headline", title),
        "descriptions": content.get("descriptions", []),
        "advantages": content.get("advantages", []),
        "characteristics": content.get("characteristics", []),
        "photos": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    resp = make_response(jsonify(payload))
    attach_session_cookie(resp, session_id, is_new)
    return resp


@app.route("/generate_photo", methods=["POST"])
def generate_photo():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()

    if not prompt:
        return jsonify({"error": "Поле 'prompt' обязательно (краткое описание товара для фото)"}), 400

    variants = []
    errors = []

    for style_key, style_suffix in PHOTO_STYLES.items():
        full_prompt = f"{prompt}, {style_suffix}"
        try:
            image = generate_image(full_prompt)
            variants.append(
                {"style": style_key, "label": PHOTO_STYLE_LABELS[style_key], "image": image}
            )
        except Exception as e:
            errors.append(f"{PHOTO_STYLE_LABELS[style_key]}: {e}")

    if not variants:
        return (
            jsonify(
                {
                    "error": "Не удалось сгенерировать ни одного изображения. "
                    + " | ".join(errors)
                }
            ),
            502,
        )

    return jsonify({"variants": variants, "errors": errors})


@app.route("/save_card", methods=["POST"])
def save_card():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "Без названия").strip()
    full_data = data.get("full_data")

    if full_data is None:
        return jsonify({"error": "Поле 'full_data' обязательно"}), 400

    try:
        full_data_str = json.dumps(full_data, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Некорректный формат full_data: {e}"}), 400

    session_id, is_new = get_session_id()
    created_at = datetime.now(timezone.utc).isoformat()

    db = get_db()
    cur = db.execute(
        "INSERT INTO cards (session_id, title, full_data, created_at) VALUES (?, ?, ?, ?)",
        (session_id, title, full_data_str, created_at),
    )
    db.commit()

    resp = make_response(
        jsonify({"id": cur.lastrowid, "title": title, "created_at": created_at})
    )
    attach_session_cookie(resp, session_id, is_new)
    return resp


@app.route("/get_cards_history", methods=["GET"])
def get_cards_history():
    session_id, is_new = get_session_id()

    db = get_db()
    rows = db.execute(
        "SELECT id, title, full_data, created_at FROM cards "
        "WHERE session_id = ? ORDER BY id DESC LIMIT 10",
        (session_id,),
    ).fetchall()

    items = []
    for row in rows:
        item = dict(row)
        try:
            item["full_data"] = json.loads(item["full_data"])
        except (TypeError, ValueError):
            item["full_data"] = None
        items.append(item)

    resp = make_response(jsonify(items))
    attach_session_cookie(resp, session_id, is_new)
    return resp


@app.route("/delete_card", methods=["POST"])
def delete_card():
    data = request.get_json(silent=True) or {}
    card_id = data.get("id")

    if not card_id:
        return jsonify({"error": "Поле 'id' обязательно"}), 400

    session_id, is_new = get_session_id()

    db = get_db()
    db.execute(
        "DELETE FROM cards WHERE id = ? AND session_id = ?",
        (card_id, session_id),
    )
    db.commit()

    resp = make_response(jsonify({"deleted": True, "id": card_id}))
    attach_session_cookie(resp, session_id, is_new)
    return resp


@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    """The primary PDF export happens fully client-side (html2canvas + jsPDF,
    see index.html) so no server-side PDF library is required. This route is
    a lightweight fallback: given the card's full_data it returns a clean,
    print-ready HTML snippet the browser can open in a new tab and print /
    'Save as PDF' from, useful if html2canvas ever fails on a given browser.
    """
    data = request.get_json(silent=True) or {}
    full_data = data.get("full_data") or {}

    headline = full_data.get("headline") or full_data.get("title") or ""
    description = ""
    descriptions = full_data.get("descriptions") or []
    if descriptions:
        description = descriptions[0]
    advantages = full_data.get("advantages") or []
    characteristics = full_data.get("characteristics") or []
    photos = full_data.get("photos") or []
    photo_src = photos[0] if photos else ""

    def esc(s):
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    advantages_html = "".join(f"<li>{esc(a)}</li>" for a in advantages)
    rows_html = "".join(
        f"<tr><td>{esc(c.get('name',''))}</td><td>{esc(c.get('value',''))}</td></tr>"
        for c in characteristics
    )
    photo_html = f'<img src="{photo_src}" style="max-width:100%;border-radius:8px;" />' if photo_src else ""

    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="UTF-8"><title>{esc(headline)}</title>
<style>
  body {{ font-family: Arial, sans-serif; padding: 32px; color: #222; }}
  h1 {{ font-size: 24px; }}
  table {{ border-collapse: collapse; margin-top: 16px; width: 100%; }}
  td {{ border: 1px solid #ccc; padding: 6px 10px; font-size: 13px; }}
  ul {{ padding-left: 20px; }}
</style></head>
<body>
  {photo_html}
  <h1>{esc(headline)}</h1>
  <p>{esc(description)}</p>
  <ul>{advantages_html}</ul>
  <table>{rows_html}</table>
  <script>window.onload = () => window.print();</script>
</body></html>"""

    resp = make_response(html)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
