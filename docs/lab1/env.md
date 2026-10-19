# Середовище стенду

- Дата: 2026-10-19
- ОС: macOS-26.5.2-x86_64-i386-64bit-Mach-O
- Python: 3.13.0
- RAM: 16.0 ГБ
- Локальний рушій: Ollama (http://localhost:11434/v1)
- Модель: qwen3:4b
- Контекст: 16384
- SDK: openai 3.35.1 · google-genai 2.31.0 · pydantic 2.14.1 · tiktoken 0.14.0 · python-dotenv 1.2.4

| # | Перевірка | Стан | Деталі |
|---|---|---|---|
| 1 | Python ≥ 3.13 | OK | 3.13.0 |
| 2 | git у PATH | OK | git version 2.33.0 |
| 3 | import bench | OK | імпортується |
| 4 | .env існує | OK | поруч із pyproject.toml; шаблон — .env.example |
| 5 | хмарний ключ | FIX | GEMINI_API_KEY: не задано · OPENROUTER_API_KEY: не задано · OPENROUTER_MODEL: не задано |
| 6 | локальний сервер і модель | FIX | Ollama (http://localhost:11434/v1) · помилка URLError |
| 7 | контекст ≥ 16384 | FIX | LOCAL_NUM_CTX=16384 · OLLAMA_CONTEXT_LENGTH=не задано |
| 8 | канарка контексту | — | запустіть з --live |
| 9 | хмарний виклик | — | запустіть з --live |
