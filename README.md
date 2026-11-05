# LLM Bench · вибір оперативного сховища

Лабораторна порівнює локальну `qwen3:4b` і Gemini на задачі вибору PostgreSQL, MongoDB, Redis або Neo4j за описом даних, навантаження й вимог.

Головна таблиця — абляція промптів у `docs/lab1/prompts.md`. У чистому клоні вона відтворюється трьома командами:

```bash
uv sync --locked
uv run pytest
uv run python scripts/report.py && git diff --exit-code docs/lab1
```

Повний звіт і статус критеріїв: [`docs/lab1/README.md`](docs/lab1/README.md).

Хмарні прогони потребують власного `GEMINI_API_KEY` у незакоміченому `.env`; для відтворення вже збережених таблиць ключ і мережа не потрібні.
