# 🏦 CAMS Voice & Chat Assistant

AI-powered multilingual voice and chat assistant for mutual fund customer support — built for **CAMS (Computer Age Management Services)**.

Supports **English · हिंदी · தமிழ்** via voice or text.

---

## Demo

https://github.com/rahulrajrr/CAMS-Voice-AI-Assistant/blob/main/assets/Demo%20Video%20.mp4

---

## System Architecture

https://github.com/rahulrajrr/CAMS-Voice-AI-Assistant/blob/main/assets/Architecture%20Diagram.jpg

> Full pipeline: Voice / Chat Input → Audio Processing → ASR → Retrieval → NLP → Action Engine → TTS → Response

---

## What it does

- 🎙️ Voice input — speak your query, get a spoken response
- 💬 Text chat — ask about portfolio, SIP, redemptions, KYC
- 📊 Portfolio queries — balance, NAV, gain/loss (enter your PAN to start)
- ⚖️ Compliance answers — SEBI rules, exit load, capital gains tax, AML/FATCA

---

## Prerequisites

- Python **3.9** (exactly) — [download here](https://www.python.org/downloads/release/python-3913/)
- FFmpeg — for audio processing

**Install FFmpeg on Windows:**
```bash
winget install --id=Gyan.FFmpeg -e
```

**Install FFmpeg on macOS:**
```bash
brew install ffmpeg
```

**Install FFmpeg on Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

---

## Get API Keys (both are free)

**Groq (LLM):**
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up → go to **API Keys** → click **Create API Key**
3. Copy the key

**Sarvam AI (Speech-to-Text + Text-to-Speech):**
1. Go to [sarvam.ai](https://sarvam.ai)
2. Sign up → go to **Dashboard** → click **Get API Key**
3. Copy the key

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/rahulrajrr/CAMS-Voice-AI-Assistant.git
cd CAMS-Voice-AI-Assistant
```

**2. Create and activate virtual environment**
```bash
# Windows
py -3.9 -m venv venv
venv\Scripts\activate.bat

# macOS / Linux
python3.9 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Create your `.env` file**

Create a file named `.env` in the project root and add your keys:
```
SARVAM_API_KEY=your_sarvam_key_here
GROQ_API_KEY=your_groq_key_here
```

**5. Generate data (run once)**
```bash
python setup_data.py
```
This creates 20 synthetic investor records and the CAMS knowledge base locally.

---

## Run

Open **two terminals**, both with the virtual environment activated.

**Terminal 1 — Backend:**
```bash
python main.py
```
API runs at `http://localhost:8000`

**Terminal 2 — Frontend:**
```bash
streamlit run streamlit_app.py
```
UI opens at `http://localhost:8501`

---

## Try it out

Once running, open `http://localhost:8501` and try:

```
hi
```
```
my PAN is ABCDE1234F, what is my balance?
```
```
what is my SIP amount?       ← no need to repeat PAN
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq `llama-3.1-8b-instant` |
| Speech-to-Text | Sarvam AI Saaras:v3 |
| Text-to-Speech | Sarvam AI Bulbul:v3 (Rahul voice) |
| Voice Activity Detection | Silero-VAD |
| Backend | FastAPI + uvicorn |
| Frontend | Streamlit |
| Vector Database | ChromaDB |

---

## Project Structure

```
├── main.py                  # FastAPI backend
├── streamlit_app.py         # Streamlit frontend
├── setup_data.py            # Generate synthetic data (run once)
├── config.py                # Settings from .env
├── schemas.py               # Request/response models
├── action_engine.py         # Compliance-first action dispatcher
├── services/
│   ├── groq_llm.py          # LLM — intent + response
│   ├── sarvam_stt.py        # Speech to text
│   ├── sarvam_tts.py        # Text to speech
│   ├── audio_processor.py   # Noise reduction + VAD
│   ├── vector_store.py      # TF-IDF knowledge search
│   └── data_retriever.py    # Investor data lookup
└── data/
    ├── synthetic_investors.py       # Generates fake investor records
```

---

## Note

Investor data is **synthetically generated** for demo purposes — no real CAMS data is used.
Backend action handlers (redemption, KYC) are stubbed with placeholders for real API integration.