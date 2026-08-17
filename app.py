import os
import sqlite3
import uuid
from datetime import datetime, timezone

import requests
from flask import Flask, request, jsonify, g, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")

# Hugging Face Inference API. Get a FREE token at https://huggingface.co/settings/tokens
# and set it as the HF_API_KEY environment variable on your hosting platform.
# The app will still run without a key, but /generate will return an error
# telling you to configure one (Hugging Face requires auth for inference calls).
HF_API_KEY = os.environ.get("HF_API_KEY", "")
HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

app = Flask(__name__, static_folder=None)
CORS(app)  # allow requests from any origin (frontend can be hosted separately)


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
    conn.commit()
    conn.close()


init_db()


# --------------------------------------------------------------------------
# Hugging Face call
# --------------------------------------------------------------------------
def call_huggingface(prompt: str) -> str:
    if not HF_API_KEY:
        raise RuntimeError(
            "HF_API_KEY не задан. Получите бесплатный токен на "
            "https://huggingface.co/settings/tokens и добавьте его как "
            "переменную окружения HF_API_KEY."
        )

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    instruct_prompt = (
        "<s>[INST] Ты — маркетолог-копирайтер. Напиши короткий продающий "
        f"пост на основе следующего описания товара: {prompt} [/INST]"
    )
    payload = {
        "inputs": instruct_prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }

    resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"HuggingFace API error {resp.status_code}: {resp.text[:300]}")

    data = resp.json()

    # HF text-generation responses are usually a list of {"generated_text": "..."}
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"HuggingFace API error: {data['error']}")

    return str(data)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    # Serves index.html if it sits next to app.py (useful for single-service deploy)
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/ping")
def ping():
    # Hit this endpoint from UptimeRobot (or similar) every 5-10 minutes
    # to keep a free Render web service from spinning down.
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
