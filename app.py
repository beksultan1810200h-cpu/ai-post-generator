import os
import re
import io
import html
import json
import time
import random
import string
import sqlite3
import uuid
from datetime import datetime, timezone

import requests
from flask import Flask, request, jsonify, g, send_from_directory, Response
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")
STATIC_DIR = os.path.join(BASE_DIR, "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Text generation via Hugging Face Inference Providers (OpenAI-compatible router).
# Get a FREE token at https://huggingface.co/settings/tokens
HF_API_KEY = os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN") or ""
HF_MODEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_TEXT_API_URL = "https://router.huggingface.co/v1/chat/completions"

# Image generation: Replicate (better quality, needs paid-ish token with free
# trial credits) with a free Hugging Face fallback.
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
REPLICATE_MODEL = os.environ.get("REPLICATE_MODEL", "black-forest-labs/flux-schnell")
HF_IMAGE_MODEL = os.environ.get("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-2-1")
HF_IMAGE_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}"

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
CORS(app, supports_credentials=True)

SESSION_COOKIE_NAME = "mp_session_id"

# --------------------------------------------------------------------------
# Category knowledge base
# --------------------------------------------------------------------------
CATEGORIES = [
    "Одежда", "Электроника", "Дом", "Детские товары",
    "Косметика", "Продукты", "Спорт", "Авто",
]

CATEGORY_CHAR_HINTS = {
    "Одежда": ["Материал", "Размер", "Цвет", "Сезон", "Страна производства", "Состав", "Уход", "Крой", "Плотность ткани", "Пол"],
    "Электроника": ["Процессор", "Оперативная память", "Встроенная память", "Диагональ экрана", "Разрешение экрана", "Вес", "Аккумулятор", "Порты", "Операционная система", "Гарантия"],
    "Дом": ["Материал", "Размеры (ДхШхВ)", "Цвет", "Вес", "Страна производства", "Уход за изделием", "Комплектация", "Стиль", "Назначение", "Гарантия"],
    "Детские товары": ["Возраст", "Материал", "Сертификат безопасности", "Размер", "Цвет", "Страна производства", "Вес", "Комплектация", "Пол", "Особенности"],
    "Косметика": ["Объём", "Тип кожи/волос", "Состав", "Страна производства", "Срок годности", "Способ применения", "Назначение", "Аромат", "Текстура", "Сертификация"],
    "Продукты": ["Вес/Объём", "Состав", "Срок годности", "Условия хранения", "Пищевая ценность", "Страна производства", "Калорийность", "Упаковка", "Вкус", "Особенности"],
    "Спорт": ["Материал", "Размер", "Вес", "Назначение", "Уровень нагрузки", "Комплектация", "Цвет", "Страна производства", "Сезон", "Гарантия"],
    "Авто": ["Совместимость (марки авто)", "Материал", "Размеры", "Вес", "Артикул производителя", "Страна производства", "Комплектация", "Назначение", "Цвет", "Гарантия"],
}
CATEGORY_CHAR_HINTS_DEFAULT = ["Материал", "Размеры", "Цвет", "Вес", "Страна производства", "Комплектация", "Гарантия", "Назначение", "Бренд", "Артикул"]

CATEGORY_MP_MAP = {
    "Одежда": {"WB": "Женщинам / Одежда", "OZON": "Одежда, обувь и аксессуары", "YM": "Одежда и обувь"},
    "Электроника": {"WB": "Электроника", "OZON": "Электроника", "YM": "Электроника"},
    "Дом": {"WB": "Дом", "OZON": "Дом и сад", "YM": "Дом и дача"},
    "Детские товары": {"WB": "Детям", "OZON": "Детские товары", "YM": "Детские товары"},
    "Косметика": {"WB": "Красота", "OZON": "Красота и здоровье", "YM": "Красота и здоровье"},
    "Продукты": {"WB": "Продукты", "OZON": "Продукты питания", "YM": "Продукты"},
    "Спорт": {"WB": "Спорт", "OZON": "Спорттовары", "YM": "Спорт и отдых"},
    "Авто": {"WB": "Автотовары", "OZON": "Автотовары", "YM": "Автотовары"},
}

PROMO_RATE_BY_CATEGORY = {
    "Одежда": 0.12, "Электроника": 0.07, "Дом": 0.10, "Детские товары": 0.11,
    "Косметика": 0.13, "Продукты": 0.09, "Спорт": 0.10, "Авто": 0.08,
}

CLOTHING_LIKE = {"Одежда", "Спорт"}
HOME_LIKE = {"Дом"}


# --------------------------------------------------------------------------
# DB helpers
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
        CREATE TABLE IF NOT EXISTS cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            marketplace TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT,
            full_data TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


# --------------------------------------------------------------------------
# Session (cookie-based, no Flask-Login needed)
# --------------------------------------------------------------------------
@app.before_request
def ensure_session():
    sid = request.cookies.get(SESSION_COOKIE_NAME)
    g.session_id = sid or uuid.uuid4().hex
    g.new_session = sid is None


@app.after_request
def attach_session_cookie(resp):
    if getattr(g, "new_session", False):
        resp.set_cookie(
            SESSION_COOKIE_NAME, g.session_id,
            max_age=60 * 60 * 24 * 365, httponly=True, samesite="Lax",
        )
    return resp


# --------------------------------------------------------------------------
# Hugging Face text generation
# --------------------------------------------------------------------------
def call_huggingface_chat(system_prompt: str, user_prompt: str, max_tokens: int = 1600) -> str:
    if not HF_API_KEY:
        raise RuntimeError(
            "HF_API_KEY не задан. Получите бесплатный токен на "
            "https://huggingface.co/settings/tokens и добавьте его как "
            "переменную окружения HF_API_KEY."
        )
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "stream": False,
    }
    resp = requests.post(HF_TEXT_API_URL, headers=headers, json=payload, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(f"HuggingFace API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"HuggingFace API error: {data['error']}")
        raise RuntimeError(f"Неожиданный формат ответа HuggingFace: {str(data)[:300]}")


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


SYSTEM_PROMPT_CARD = (
    "Ты — профессиональный копирайтер и SEO-специалист маркетплейсов Wildberries, "
    "Ozon и Яндекс.Маркет. Отвечай СТРОГО в формате JSON без markdown-разметки, "
    "без ```json, без пояснений до или после — только валидный JSON-объект."
)


def build_card_text_prompt(title, category, brand, price, description, advantages_raw, hints):
    hints_str = ", ".join(hints)
    return f"""Товар: {title}
Категория: {category}
Бренд: {brand or "не указан"}
Цена: {price or "не указана"} руб.
Описание от продавца: {description}
Ключевые преимущества от продавца: {advantages_raw or "не указаны"}

Сгенерируй карточку товара для маркетплейсов. Характеристики подбирай под категорию
"{category}", ориентируйся на параметры: {hints_str} (допустимо заменить похожими,
если не подходят к товару).

Верни ТОЛЬКО JSON со следующей структурой:
{{
  "title_wb": "название товара до 40 символов",
  "title_ozon": "название товара до 60 символов",
  "title_ym": "название товара до 60 символов",
  "description_short": "короткое описание, 1-2 предложения",
  "description_medium": "среднее SEO-описание, 3-5 предложений",
  "description_long": "расширенное описание для бренд-зоны, 6-10 предложений",
  "characteristics": [{{"param": "название параметра", "value": "значение"}}],
  "advantages": ["преимущество с эмодзи в начале"],
  "seo_keywords": {{"wb": ["слово"], "ozon": ["слово"], "ym": ["слово"]}}
}}
В characteristics должно быть ровно 10 пунктов, в advantages ровно 5,
в каждом списке seo_keywords ровно 10 слов. Пиши на русском языке."""


# --------------------------------------------------------------------------
# Image generation (Replicate primary, Hugging Face free fallback)
# --------------------------------------------------------------------------
def save_image(img_bytes: bytes, ext: str = "png") -> str:
    filename = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(IMAGES_DIR, filename)
    with open(path, "wb") as f:
        f.write(img_bytes)
    return f"/static/images/{filename}"


def call_replicate_image(prompt: str) -> bytes:
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN не задан")
    url = f"https://api.replicate.com/v1/models/{REPLICATE_MODEL}/predictions"
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait=60",
    }
    resp = requests.post(url, headers=headers, json={"input": {"prompt": prompt}}, timeout=70)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Replicate API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    status = data.get("status")
    output = data.get("output")
    get_url = (data.get("urls") or {}).get("get")
    attempts = 0
    while status not in ("succeeded", "failed", "canceled") and get_url and attempts < 20:
        time.sleep(3)
        r2 = requests.get(get_url, headers=headers, timeout=30)
        d2 = r2.json()
        status = d2.get("status")
        output = d2.get("output")
        attempts += 1
    if status != "succeeded" or not output:
        raise RuntimeError(f"Replicate: генерация не удалась (status={status})")
    img_url = output[0] if isinstance(output, list) else output
    img_resp = requests.get(img_url, timeout=60)
    if img_resp.status_code != 200:
        raise RuntimeError("Не удалось скачать изображение с Replicate")
    return img_resp.content


def call_hf_image(prompt: str) -> bytes:
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY не задан")
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    resp = requests.post(HF_IMAGE_API_URL, headers=headers, json={"inputs": prompt}, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HF Image API error {resp.status_code}: {resp.text[:300]}")
    if "image" not in resp.headers.get("Content-Type", ""):
        raise RuntimeError(f"HF Image API: неожиданный ответ ({resp.text[:200]})")
    return resp.content


def generate_single_image(prompt: str):
    errors = []
    if REPLICATE_API_TOKEN:
        try:
            return save_image(call_replicate_image(prompt)), None
        except Exception as e:
            errors.append(f"Replicate: {e}")
    try:
        return save_image(call_hf_image(prompt)), None
    except Exception as e:
        errors.append(f"HuggingFace: {e}")
    return None, "; ".join(errors)


def build_photo_prompt(title, description, keywords, category, photo_type):
    base = f"Professional e-commerce product photography of {title}. {description[:150]}. Keywords: {keywords}."
    variants = {
        "studio": " Studio shot, pure white background, soft even lighting, high resolution, centered composition, no text, no watermark.",
        "model": " Photo of a model wearing/using the product, natural lighting, lifestyle fashion photography, neutral background, no text, no watermark.",
        "interior": " Product placed in a modern cozy interior room setting, realistic lighting, home decor photography, no text, no watermark.",
        "closeup": " Extreme close-up macro shot showing product details and texture, sharp focus, studio lighting, no text, no watermark.",
    }
    return base + variants.get(photo_type, variants["studio"])


# --------------------------------------------------------------------------
# Marketplace helpers
# --------------------------------------------------------------------------
def generate_article(marketplace: str) -> str:
    prefix = {"WB": "WB", "OZON": "OZ", "YM": "YM"}.get(marketplace, "MP")
    number = "".join(random.choices(string.digits, k=8))
    return f"{prefix}-{number}"


def estimate_promo_cost(price, category):
    try:
        price_val = float(price)
    except (TypeError, ValueError):
        return {"estimated_daily_budget": None, "note": "Укажите цену, чтобы увидеть ориентировочный расчёт."}
    rate = PROMO_RATE_BY_CATEGORY.get(category, 0.10)
    estimate = round(price_val * rate, 2)
    return {
        "estimated_daily_budget": estimate,
        "note": "Ориентировочный расчёт (цена × средний % на продвижение по категории). "
                "Не является официальной ставкой площадки.",
    }


def esc(s):
    return html.escape(str(s if s is not None else ""))


# --------------------------------------------------------------------------
# Routes: pages
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/ping")
def ping():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})


# --------------------------------------------------------------------------
# Routes: card content generation
# --------------------------------------------------------------------------
@app.route("/generate_card", methods=["POST"])
def generate_card():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    category = (data.get("category") or "").strip()
    brand = (data.get("brand") or "").strip()
    price = data.get("price")
    description = (data.get("description") or "").strip()
    advantages_raw = (data.get("advantages") or "").strip()

    if not title or not category or not description:
        return jsonify({"error": "Заполните название, категорию и описание"}), 400

    hints = CATEGORY_CHAR_HINTS.get(category, CATEGORY_CHAR_HINTS_DEFAULT)
    prompt = build_card_text_prompt(title, category, brand, price, description, advantages_raw, hints)

    try:
        raw = call_huggingface_chat(SYSTEM_PROMPT_CARD, prompt)
        content = extract_json(raw)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Сетевая ошибка при обращении к HuggingFace: {e}"}), 502
    except (ValueError, json.JSONDecodeError):
        return jsonify({"error": "Не удалось разобрать ответ ИИ как JSON. Попробуйте ещё раз."}), 502

    articles = {mp: generate_article(mp) for mp in ("WB", "OZON", "YM")}
    suggested_categories = CATEGORY_MP_MAP.get(category, {})
    promo_estimate = estimate_promo_cost(price, category)

    return jsonify({
        "title": title,
        "brand": brand,
        "category": category,
        "price": price,
        "content": content,
        "suggested_categories": suggested_categories,
        "articles": articles,
        "promo_estimate": promo_estimate,
    })


@app.route("/generate_photo", methods=["POST"])
def generate_photo():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()
    keywords = (data.get("keywords") or "").strip()
    category = (data.get("category") or "").strip()
    requested_types = data.get("types")

    if not title:
        return jsonify({"error": "Укажите название товара"}), 400

    if requested_types:
        types = requested_types
    else:
        types = ["studio", "studio2", "closeup"]
        if category in CLOTHING_LIKE:
            types.append("model")
        if category in HOME_LIKE:
            types.append("interior")

    images, errors = {}, {}
    for t in types:
        base_type = t.replace("2", "")
        prompt = build_photo_prompt(title, description, keywords, category, base_type)
        url, err = generate_single_image(prompt)
        if url:
            images.setdefault(base_type, []).append(url)
        if err:
            errors[t] = err

    if not images:
        return jsonify({"error": "Не удалось сгенерировать ни одного изображения", "details": errors}), 502

    return jsonify({"images": images, "errors": errors})


# --------------------------------------------------------------------------
# Routes: persistence
# --------------------------------------------------------------------------
@app.route("/save_card", methods=["POST"])
def save_card():
    data = request.get_json(silent=True) or {}
    marketplace = (data.get("marketplace") or "WB").upper()
    title = data.get("title") or "Без названия"
    category = data.get("category") or ""
    full_data = json.dumps(data, ensure_ascii=False)

    db = get_db()
    cur = db.execute(
        "INSERT INTO cards (session_id, marketplace, title, category, full_data, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (g.session_id, marketplace, title, category, full_data, datetime.now(timezone.utc).isoformat()),
    )
    db.commit()
    return jsonify({"id": cur.lastrowid, "status": "saved"})


@app.route("/get_history", methods=["GET"])
def get_history():
    marketplace = (request.args.get("marketplace") or "").strip().upper()
    db = get_db()
    if marketplace and marketplace != "ALL":
        rows = db.execute(
            "SELECT id, marketplace, title, category, full_data, created_at FROM cards "
            "WHERE session_id = ? AND marketplace = ? ORDER BY id DESC LIMIT 15",
            (g.session_id, marketplace),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT id, marketplace, title, category, full_data, created_at FROM cards "
            "WHERE session_id = ? ORDER BY id DESC LIMIT 15",
            (g.session_id,),
        ).fetchall()

    result = []
    for r in rows:
        thumbnail = None
        try:
            fd = json.loads(r["full_data"])
            for v in (fd.get("photos") or {}).values():
                if isinstance(v, list) and v:
                    thumbnail = v[0]
                    break
        except Exception:
            pass
        result.append({
            "id": r["id"], "marketplace": r["marketplace"], "title": r["title"],
            "category": r["category"], "created_at": r["created_at"], "thumbnail": thumbnail,
        })
    return jsonify(result)


@app.route("/get_card/<int:card_id>", methods=["GET"])
def get_card(card_id):
    db = get_db()
    row = db.execute(
        "SELECT full_data FROM cards WHERE id = ? AND session_id = ?", (card_id, g.session_id)
    ).fetchone()
    if not row:
        return jsonify({"error": "Карточка не найдена"}), 404
    return jsonify(json.loads(row["full_data"]))


@app.route("/delete_card", methods=["POST"])
def delete_card():
    data = request.get_json(silent=True) or {}
    card_id = data.get("id")
    db = get_db()
    db.execute("DELETE FROM cards WHERE id = ? AND session_id = ?", (card_id, g.session_id))
    db.commit()
    return jsonify({"status": "deleted"})


# --------------------------------------------------------------------------
# Routes: export
# --------------------------------------------------------------------------
def render_print_html(data: dict) -> str:
    content = data.get("content") or {}
    title = data.get("title") or "Товар"
    brand = data.get("brand") or ""
    price = data.get("price") or ""
    photos = data.get("photos") or {}

    first_photo = ""
    for v in photos.values():
        if isinstance(v, list) and v:
            first_photo = v[0]
            break

    chars_rows = "".join(
        f"<tr><td>{esc(c.get('param',''))}</td><td>{esc(c.get('value',''))}</td></tr>"
        for c in (content.get("characteristics") or [])
    )
    adv_items = "".join(f"<li>{esc(a)}</li>" for a in (content.get("advantages") or []))
    description = content.get("description_medium") or content.get("description_short") or ""

    return f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="UTF-8"><title>{esc(title)} — презентация</title>
<style>
 body{{font-family:Arial,sans-serif;color:#222;padding:40px;max-width:800px;margin:0 auto;}}
 h1{{font-size:26px;margin-bottom:4px;}}
 .brand{{color:#888;margin-bottom:20px;}}
 .price{{font-size:22px;font-weight:bold;color:#e63946;margin-bottom:20px;}}
 img.photo{{max-width:100%;border-radius:12px;margin-bottom:20px;}}
 table{{width:100%;border-collapse:collapse;margin-bottom:20px;}}
 td{{border:1px solid #ddd;padding:8px;font-size:14px;}}
 ul{{padding-left:20px;}}
 li{{margin-bottom:6px;}}
 @media print {{ body{{padding:0;}} }}
</style></head>
<body>
<h1>{esc(title)}</h1>
<div class="brand">{esc(brand)}</div>
{f'<div class="price">{esc(str(price))} ₽</div>' if price else ''}
{f'<img class="photo" src="{esc(first_photo)}">' if first_photo else ''}
<p>{esc(description)}</p>
<h3>Характеристики</h3>
<table>{chars_rows}</table>
<h3>Преимущества</h3>
<ul>{adv_items}</ul>
<script>window.onload = function(){{ setTimeout(function(){{ window.print(); }}, 300); }};</script>
</body></html>"""


@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    data = request.get_json(silent=True) or {}
    return Response(render_print_html(data), mimetype="text/html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
