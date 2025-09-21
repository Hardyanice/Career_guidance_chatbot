import streamlit as st
import os, json, faiss, joblib
import numpy as np
import cohere, PyPDF2, spacy
from sklearn.feature_extraction.text import CountVectorizer
import traceback

# --- CONFIG ---
COHERE_API_KEY = "PgSVyIVx2nYWXthLEq9mZpdOy1fG7FgQFpakeLvG"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Cohere client with error handling
try:
    co = cohere.Client(COHERE_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Cohere client: {e}")
    st.stop()

# Load spaCy model with error handling
def install_local_model():
    whl_file = "en_core_web_sm-3.8.0-py3-none-any.whl"
    
    try:
        # Install the local .whl file
        subprocess.run([sys.executable, "-m", "pip", "install", whl_file], 
                      check=True, capture_output=True)
        print("✅ Model installed successfully!")
        
        # Now load it normally
        nlp = spacy.load("en_core_web_sm")
        return nlp
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing model: {e}")
        raise
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

# Use it
nlp = install_local_model()

# --- Load Data with error handling ---
@st.cache_data
def load_faq_data():
    try:
        with open("faq.json", "r") as f:
            faq_data = json.load(f)
        return [f"Question: {x['question']}\nAnswer: {x['answer']}" for x in faq_data]
    except FileNotFoundError:
        st.error("faq.json file not found. Please ensure the file exists.")
        return []
    except Exception as e:
        st.error(f"Error loading FAQ data: {e}")
        return []

@st.cache_data
def load_intent_data():
    try:
        with open("intent_detection.json", "r") as f:
            kb = json.load(f)
        return kb, [x["prompt"] for x in kb]
    except FileNotFoundError:
        st.error("intent_detection.json file not found. Please ensure the file exists.")
        return [], []
    except Exception as e:
        st.error(f"Error loading intent data: {e}")
        return [], []

@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load("vectorizer.joblib")
        classifier = joblib.load("classifier.joblib")
        return vectorizer, classifier
    except FileNotFoundError:
        st.error("Model files (vectorizer.joblib, classifier.joblib) not found.")
        return None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load data
faq_docs = load_faq_data()
kb, intents = load_intent_data()
vectorizer, classifier = load_models()

def embed(texts, input_type="search_document"):
    try:
        if not texts or len(texts) == 0:
            return np.array([])
        
        # Use the v3.0 model with proper response handling
        response = co.embed(
            model="embed-english-v3.0", 
            texts=texts,
            input_type=input_type,
            embedding_types=["float"]
        )
        
        # Handle the new response structure for v3.0
        if hasattr(response, 'embeddings') and hasattr(response.embeddings, 'float'):
            # New v3.0 structure: response.embeddings.float
            embeddings = response.embeddings.float
        elif hasattr(response, 'embeddings'):
            # Fallback: direct embeddings access
            embeddings = response.embeddings
        else:
            # Last resort: try to access embeddings differently
            embeddings = getattr(response, 'embeddings', [])
        
        return np.array(embeddings).astype("float32")
                
    except Exception as e:
        # Try fallback with v2.0 model without input_type
        try:
            response = co.embed(model="embed-english-v2.0", texts=texts)
            return np.array(response.embeddings).astype("float32")
        except Exception as e2:
            st.error(f"Error generating embeddings: {e}")
            return np.array([])

# --- Initialize indexes with error handling ---
@st.cache_resource
def initialize_faq_index():
    if not faq_docs:
        return None, None
    try:
        faq_emb = embed(faq_docs, input_type="search_document")
        if faq_emb.size == 0:
            return None, None
        faq_index = faiss.IndexFlatL2(faq_emb.shape[1])
        faq_index.add(faq_emb)
        return faq_index, faq_emb
    except Exception as e:
        st.error(f"Error initializing FAQ index: {e}")
        return None, None

@st.cache_resource
def initialize_intent_index():
    if not intents:
        return None, None
    try:
        intent_emb = embed(intents, input_type="search_document")
        if intent_emb.size == 0:
            return None, None
        intent_emb = intent_emb / np.linalg.norm(intent_emb, axis=1, keepdims=True)
        intent_index = faiss.IndexFlatIP(intent_emb.shape[1])
        intent_index.add(intent_emb)
        return intent_index, intent_emb
    except Exception as e:
        st.error(f"Error initializing intent index: {e}")
        return None, None

# Initialize indexes
faq_result = initialize_faq_index()
intent_result = initialize_intent_index()

faq_index = faq_result[0] if faq_result else None
intent_index = intent_result[0] if intent_result else None

# --- Core functions ---
def faq_module(user_input):
    try:
        if not faq_index:
            return "FAQ system is not available."
        
        q = embed([user_input], input_type="search_query")
        if q.size == 0:
            return "Sorry, I couldn't process your question."
        
        _, idx = faq_index.search(q, 1)
        retrieved = faq_docs[idx[0][0]]
        
        prompt = f"Context:\n{retrieved}\n\nUser: {user_input}\nAnswer:"
        response = co.chat(
            model="command-xlarge-nightly",
            message=prompt,
            max_tokens=300,
            temperature=0.6
        )
        return response.text.strip()
    except Exception as e:
        return f"Error processing FAQ: {str(e)}"

def predict_intent(user_input):
    try:
        if not intent_index or not kb:
            return "general_chat", 0.0
        
        q = embed([user_input], input_type="search_query")
        if q.size == 0:
            return "general_chat", 0.0
        
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        dist, idx = intent_index.search(q, 1)
        return kb[idx[0][0]]["intent"], dist[0][0]
    except Exception as e:
        st.error(f"Error predicting intent: {e}")
        return "general_chat", 0.0

def extract_text(pdf):
    try:
        reader = PyPDF2.PdfReader(pdf)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def resume_classifier_module(pdf):
    try:
        if not classifier or not vectorizer:
            return "Resume classifier is not available."
        
        txt = extract_text(pdf)
        if not txt.strip():
            return "Could not extract text from the PDF."
        
        doc = nlp(txt.lower())
        tokens = " ".join([t.lemma_ for t in doc if t.is_alpha and not t.is_stop])
        
        if not tokens.strip():
            return "No meaningful text found in the resume."
        
        X = vectorizer.transform([tokens])
        prediction = classifier.predict(X)[0]
        
        prompt = f"You are an expert career coach. Explain why this candidate is suited for '{prediction}' in 5 concise points.\nResume:\n{txt[:2000]}..."  # Limit text length
        
        response = co.chat(
            model='command-xlarge-nightly',
            message=prompt,
            max_tokens=300,
            temperature=0.6
        )
        return f"Predicted Role: {prediction}\n\nExplanation:\n{response.text.strip()}"
    except Exception as e:
        return f"Error in resume classification: {str(e)}"

def extract_skills(text, top_k=20):
    try:
        if not text.strip():
            return []
        
        doc = nlp(text)
        chunks = [chunk.text.lower().strip() for chunk in doc.noun_chunks if chunk.text.strip()]
        
        if not chunks:
            return []
        
        vectorizer_local = CountVectorizer(ngram_range=(1,2), stop_words='english', max_features=1000)
        vectorizer_local.fit(chunks)
        features = vectorizer_local.get_feature_names_out()
        return list(set(features))[:top_k]
    except Exception as e:
        st.error(f"Error extracting skills: {e}")
        return []

def skill_gap_module(resume_file, jd_file):
    try:
        def clean(text):
            if not text.strip():
                return []
            doc = nlp(text)
            return [t.text.lower() for t in doc if t.is_alpha and not t.is_stop]

        resume_text = extract_text(resume_file)
        jd_text = extract_text(jd_file)
        
        if not resume_text.strip() or not jd_text.strip():
            return "Could not extract text from one or both PDF files."

        resume_tokens = clean(resume_text)
        jd_tokens = clean(jd_text)

        resume_skills = extract_skills(" ".join(resume_tokens))
        jd_skills = extract_skills(" ".join(jd_tokens))

        missing_skills = list(set(jd_skills) - set(resume_skills))

        message = f"""
Resume skills: {', '.join(resume_skills[:20])}
Job requires: {', '.join(jd_skills[:20])}
Missing: {', '.join(missing_skills[:10])}
Suggest improvements for the candidate's resume:
"""
        response = co.chat(
            model='command-xlarge-nightly',
            message=message,
            temperature=0.5,
            max_tokens=400
        )
        return f"Missing Skill Keywords (Top 10): {missing_skills[:10]}\n\n{response.text.strip()}"
    except Exception as e:
        return f"Error in skill gap analysis: {str(e)}"

# --- Session state initialization ---
def init_session_state():
    if "intent" not in st.session_state:
        st.session_state.intent = None
    if "resume_file" not in st.session_state:
        st.session_state.resume_file = None
    if "jd_file" not in st.session_state:
        st.session_state.jd_file = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input_bottom_clear" not in st.session_state:
        st.session_state.user_input_bottom_clear = False

init_session_state()

# --- Streamlit UI ---
st.title("💡 AI Career Guidance Chatbot")

# Add system status
with st.expander("System Status", expanded=False):
    st.write(f"FAQ System: {'✅ Ready' if faq_index else '❌ Not Available'}")
    st.write(f"Intent Detection: {'✅ Ready' if intent_index else '❌ Not Available'}")
    st.write(f"Resume Classifier: {'✅ Ready' if classifier and vectorizer else '❌ Not Available'}")

chat_box = st.container()

# --- Display chat history ---
with chat_box:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            # User message - right aligned with grey bubble
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                <div style="background-color: #E8E8E8; padding: 12px 16px; border-radius: 18px; max-width: 70%; border-bottom-right-radius: 4px; color: #000000;">
                    <strong style="color: #000000;">You:</strong> <span style="color: #000000;">{msg['content']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # AI message - left aligned with grey bubble
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                <div style="background-color: #F0F0F0; padding: 12px 16px; border-radius: 18px; max-width: 70%; border-bottom-left-radius: 4px; color: #000000;">
                    <strong style="color: #000000;">AI:</strong> <span style="color: #000000;">{msg['content']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- Functions to handle buttons ---
def handle_resume():
    if st.session_state.resume_file:
        with st.spinner("Analyzing resume..."):
            result = resume_classifier_module(st.session_state.resume_file)
        st.session_state.chat_history.append({"role": "ai", "content": result})
    else:
        st.session_state.chat_history.append({"role": "ai", "content": "Please upload your Resume PDF for classification."})
    st.rerun()

def handle_skill_gap():
    if st.session_state.resume_file and st.session_state.jd_file:
        with st.spinner("Analyzing skill gaps..."):
            result = skill_gap_module(st.session_state.resume_file, st.session_state.jd_file)
        st.session_state.chat_history.append({"role": "ai", "content": result})
    else:
        st.session_state.chat_history.append({"role": "ai", "content": "Please upload both Resume and Job Description PDFs."})
    st.rerun()

# --- Text input ---
user_input = st.text_area(
    "Type your message...",
    value="",
    height=68,
    key="user_input_bottom",
    placeholder="Write your message..."
)

# --- Handle general chat / FAQ ---
if st.button("Send", type="primary"):
    text = user_input.strip()
    if text:
        st.session_state.chat_history.append({"role": "user", "content": text})
        
        with st.spinner("Processing..."):
            intent, sim = predict_intent(text)
            st.session_state.intent = intent if sim >= 0.5 else "general_chat"

            # Generate response
            if st.session_state.intent == "faq":
                ai_response = faq_module(text)
                st.session_state.chat_history.append({"role": "ai", "content": ai_response})
            elif st.session_state.intent == "resume_job_classifier":
                st.session_state.chat_history.append({"role": "ai", "content": "Click the button below to upload Resume PDF for classification."})
            elif st.session_state.intent == "skill_gap_analyst":
                st.session_state.chat_history.append({"role": "ai", "content": "Click the button below to upload Resume and Job Description PDFs for Skill Gap Analysis."})
            else:
                try:
                    prompt = f"Have a friendly conversation with the user.\nUser: {text}\nAI:"
                    response = co.chat(
                        model="command-xlarge-nightly",
                        message=prompt,
                        temperature=0.6,
                        max_tokens=300
                    )
                    ai_response = response.text.strip()
                    st.session_state.chat_history.append({"role": "ai", "content": ai_response})
                except Exception as e:
                    st.session_state.chat_history.append({"role": "ai", "content": f"Sorry, I encountered an error: {str(e)}"})
        
        st.rerun()

# --- File upload and buttons ---
if st.session_state.intent == "resume_job_classifier":
    st.subheader("Resume Analysis")
    pdf = st.file_uploader("Upload Resume PDF", type="pdf", key="resume_upload")
    if pdf:
        st.session_state.resume_file = pdf
        st.success("Resume uploaded successfully!")
    
    if st.session_state.resume_file:
        if st.button("✅ Get Resume Analysis", on_click=handle_resume):
            pass

if st.session_state.intent == "skill_gap_analyst":
    st.subheader("Skill Gap Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        resume = st.file_uploader("Upload Resume PDF", type="pdf", key="skill_resume")
        if resume:
            st.session_state.resume_file = resume
            st.success("Resume uploaded!")
    
    with col2:
        jd = st.file_uploader("Upload Job Description PDF", type="pdf", key="skill_jd")
        if jd:
            st.session_state.jd_file = jd
            st.success("Job Description uploaded!")
    
    if st.session_state.resume_file and st.session_state.jd_file:
        if st.button("✅ Get Skill Gap Analysis", on_click=handle_skill_gap):
            pass

# --- Sidebar ---
with st.sidebar:
    st.header("Actions")
    if st.button("🔄 Reset Conversation", type="secondary"):
        st.session_state.intent = None
        st.session_state.resume_file = None
        st.session_state.jd_file = None
        st.session_state.chat_history = []
        st.success("Conversation reset!")
        st.rerun()
    
    if st.button("🔧 Clear Cache & Reinitialize", type="secondary"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Cache cleared!")
        st.rerun()
    
    st.header("Instructions")
    st.write("""
    1. Ask questions for FAQ support
    2. Request resume analysis
    3. Ask for skill gap analysis
    4. Chat about career guidance
    """)





