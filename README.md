# hindi-qa-chatbot
# 🇮🇳 Hindi AI Chatbot 
An intelligent conversational AI chatbot that understands and responds fluently in pure Hindi (Devanagari script) and Hinglish. Built using Python, Streamlit, and modern Large Language Models (LLMs).


##  Features
* **Bilingual Understanding**: Processes inputs in Devanagari (हिंदी)
* **Natural Responses**: Generates context-aware, grammatically accurate Hindi replies.
* **Interactive UI**: Clean, responsive chat interface powered by Streamlit.
* **Chat History**: Maintains conversation context dynamically during the session.

---

##  Project Structure
```text
hindi-chatbot/
├── .streamlit/
│   └── secrets.toml      # API credentials configuration
├── app.py                # Main Streamlit application interface
├── requirements.txt      # Python package dependencies
├── .env                  # Local environment variables
└── README.md             # Project documentation
```

---

##  Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com
cd hindi-chatbot
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your LLM API key:
```env
OPENAI_API_KEY=your_actual_api_key_here
```

---

## Running the Application

Launch the Streamlit web interface locally:
```bash
streamlit run app.py
```

Open the provided local URL (usually `http://localhost:8501`) in your web browser.

---

## Requirements (`requirements.txt`)
```text
streamlit>=1.30.0
openai>=1.0.0
python-dotenv>=1.0.0
```

---

