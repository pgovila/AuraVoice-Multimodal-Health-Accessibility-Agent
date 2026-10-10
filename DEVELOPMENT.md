# DEVELOPMENT.md — AuraVoice Local Setup

Common environment issues extracted from README to keep it clean.

---

## Python Version

AuraVoice requires **Python 3.12**. Python 3.13 and 3.14 are not supported
by all dependencies (`streamlit-mic-recorder`, `SpeechRecognition`).

Check your version:
```cmd
python --version
```

If it says 3.14 or 3.13, specify 3.12 explicitly when creating the venv:

```cmd
C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe -m venv venv
```

Find your 3.12 installation:
```cmd
where python
```
Look for `Python312` in the path.

---

## Creating the Virtual Environment

### Windows (cmd)
```cmd
C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Copying the Secrets File

### Windows cmd
```cmd
copy ".streamlit\secrets.toml.example" ".streamlit\secrets.toml"
notepad ".streamlit\secrets.toml"
```

### Windows PowerShell
```powershell
Copy-Item ".streamlit\secrets.toml.example" ".streamlit\secrets.toml"
notepad ".streamlit\secrets.toml"
```

### macOS / Linux
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
nano .streamlit/secrets.toml
```

---

## Configuring Model IDs Without Code Changes

Model IDs are configurable via environment variables:

```cmd
set AURAVOICE_GEMINI_MODEL=gemini-3.6-flash
set AURAVOICE_GROQ_MODEL=qwen/qwen3.8-27b
streamlit run app.py
```

If Google or Groq retire a model, update these env vars without touching `app.py`.

Check available models:

**Gemini:**
```python
import google.generativeai as genai
genai.configure(api_key="YOUR_KEY")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
```

**Groq:**
```python
from groq import Groq
client = Groq(api_key="YOUR_KEY")
for m in client.models.list().data:
    print(m.id)
```

---

## Running in Production Mode

Set `AURAVOICE_PROD=1` to hide the API key fields from the sidebar
(keys must be in `st.secrets` or environment variables):

```cmd
set AURAVOICE_PROD=1
streamlit run app.py
```

---

## Common Install Errors

| Error | Fix |
|---|---|
| `st-audiorec` not found | Replaced by `streamlit-mic-recorder` — already in `requirements.txt` |
| `Copy-Item` not recognised in cmd.exe | Use `copy` instead (cmd command) |
| `google-generativeai` FutureWarning | Expected — we import `google.genai` not `google.generativeai` |
| `No matching distribution for streamlit-mic-recorder` | Ensure you are on Python 3.12, not 3.14 |
