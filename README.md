# rag-chatbot

## Environment / API key setup

For security, do NOT hardcode API keys in the source. You can securely provide a Google API key via environment variables or a `.env` file.

1. Local dev using a `.env` file at the repo root (not committed):
```
GOOGLE_API_KEY=your_key_here
```
Make sure `.env` is added to `.gitignore` so it does not get committed.

2. Export environment variable (macOS/Linux):
```
export GOOGLE_API_KEY=your_key_here
```

3. The app also supports entering the API key via UI temporarily for the duration of the session (not persisted to disk).

These approaches are safer than putting keys directly in source files or committing them to version control.

UI auto-save and manual save
---------------------------
The app provides a UI checkbox to automatically save the entered Google API key into a local `.env` file, and buttons to save/remove the key manually.
If you use these features, ensure `.env` is added to `.gitignore` to prevent accidental commits.
