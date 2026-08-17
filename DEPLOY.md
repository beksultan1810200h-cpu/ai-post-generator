# Развёртывание на Render.com (бесплатно)

## 1. Подготовка репозитория
1. Создай аккаунт на https://github.com (если нет).
2. Создай новый репозиторий, например `ai-post-generator`.
3. Залей в него 4 файла: `app.py`, `index.html`, `requirements.txt`, `Procfile`.
   (index.html лежит рядом с app.py — Flask отдаёт его на маршруте `/`.)

## 2. Получи бесплатный API-ключ Hugging Face
1. Зарегистрируйся на https://huggingface.co
2. Перейди в https://huggingface.co/settings/tokens
3. Нажми "New token" → тип "Read" → создай.
4. Скопируй токен (начинается с `hf_...`) — он понадобится на шаге 4.

## 3. Создай Web Service на Render
1. Зайди на https://render.com и зарегистрируйся (можно через GitHub).
2. Нажми **New +** → **Web Service**.
3. Выбери свой репозиторий `ai-post-generator`.
4. Заполни настройки:
   - **Name**: любое, например `ai-post-generator`
   - **Region**: любой ближайший
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: `Free`

## 4. Добавь переменные окружения
В разделе **Environment** (или на этапе создания сервиса) добавь:
- `HF_API_KEY` = твой токен `hf_...` из шага 2
- (опционально) `HF_MODEL` = `mistralai/Mixtral-8x7B-Instruct-v0.1` (или другая instruct-модель, доступная через Inference API)

## 5. Деплой
1. Нажми **Create Web Service**.
2. Render сам установит зависимости и запустит `gunicorn app:app`.
3. Через 1–3 минуты появится публичная ссылка вида:
   `https://ai-post-generator.onrender.com`
4. Открой её — увидишь страницу генератора.

## 6. SQLite
Файл `database.db` создаётся автоматически при первом запуске (см. `init_db()` в `app.py`).
⚠️ На бесплатном плане Render файловая система эфемерна — при передеплое или пересборке данные в `database.db` могут обнулиться. Это ожидаемо для файловой SQLite на бесплатном хостинге; для постоянного хранения в будущем можно подключить Render Disk (платно) или внешнюю БД.

## 7. Чтобы сервис не «засыпал» (Free план Render усыпляет сервис после ~15 минут простоя)
Вариант А — простой и бесплатный:
1. Зарегистрируйся на https://uptimerobot.com (бесплатно).
2. Создай монитор типа **HTTP(s)**.
3. URL: `https://твой-сервис.onrender.com/ping`
4. Интервал проверки: 5 минут.
Это будет "будить" сервис регулярным пингом.

Вариант Б — Railway.app вместо Render:
1. Зайди на https://railway.app, зарегистрируйся через GitHub.
2. **New Project** → **Deploy from GitHub repo** → выбери репозиторий.
3. В **Variables** добавь `HF_API_KEY`.
4. Railway сам определит Python-проект по `requirements.txt` и `Procfile`.
5. Бесплатный план даёт 500 часов/месяц — сервис не засыпает так агрессивно, как Render Free, но всё равно ограничен по времени/ресурсам.

## 8. Замена API-ключа своим
В `app.py` ключ читается из переменной окружения:
```python
HF_API_KEY = os.environ.get("HF_API_KEY", "")
```
Ничего в коде менять не нужно — просто задай `HF_API_KEY` в настройках Render/Railway (шаг 4). Никогда не хардкодь ключ прямо в `app.py`, если репозиторий публичный.

## 9. Локальный запуск (для проверки перед деплоем)
```bash
pip install -r requirements.txt
export HF_API_KEY=hf_твой_токен   # Windows: set HF_API_KEY=hf_твой_токен
python app.py
```
Открой http://localhost:5000
