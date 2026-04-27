import streamlit as st
import anthropic

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="MHT-CET Helpdesk Chatbot",
    page_icon="🎓",
    layout="centered"
)

# -------------------------------
# SYSTEM PROMPT
# -------------------------------
SYSTEM_PROMPT = """You are an expert MHT-CET helpdesk assistant for Maharashtra, India. You help students and parents with accurate, concise, friendly answers about:

- MHT-CET exam: eligibility, pattern, syllabus, exam dates, admit card, negative marking, scoring, normalization, PCM/PCB groups
- Application process: fees, form correction, Aadhaar, documents required
- Results: percentile, scorecard, checking results
- CAP rounds: registration, seat allotment, preferences, how many rounds, spot rounds
- College selection: COEP, VJTI, SPIT, autonomous vs non-autonomous, fees, placements
- Scholarships: government schemes, MahaDBT, EBC, OBC, SC/ST
- Reservations: TFWS seats, minority quota, caste validity, domicile certificate
- Preparation tips: books, mock tests, last-month strategy, coaching
- Related exams: JEE vs MHT-CET differences, lateral entry for diploma students
- Gap year: gap certificate, dropping a year, attempt limits

Rules:
- Keep answers concise (2-4 sentences typically), friendly, and helpful
- For very specific cutoffs, say they vary yearly and suggest checking the official CET Cell website: cetcell.mahacet.org
- If unsure, say so and recommend the official site
- Respond in English by default, but if the user writes in Hindi or Marathi, respond in that language
- Add 1-2 relevant follow-up suggestions at the end to guide the student
- Use bullet points only when listing multiple items
- Never make up exam dates or exact cutoff scores — say these vary and must be verified officially"""

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
    .stApp { max-width: 750px; margin: auto; }

    .chat-header {
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .chat-header h2 { margin: 0; font-size: 20px; }
    .chat-header p  { margin: 0; font-size: 13px; opacity: 0.85; }

    .user-bubble {
        background: #dbeafe;
        color: #1e3a8a;
        padding: 10px 15px;
        border-radius: 18px 18px 4px 18px;
        margin: 6px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 14px;
        line-height: 1.5;
    }
    .bot-bubble {
        background: #f1f5f9;
        color: #1e293b;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 4px;
        margin: 6px 0;
        max-width: 80%;
        font-size: 14px;
        line-height: 1.5;
    }
    .msg-label {
        font-size: 11px;
        color: #94a3b8;
        margin: 2px 4px;
    }
    .msg-label.user { text-align: right; }

    .source-badge {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 10px;
        margin-top: 4px;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0 16px;
    }
    .chip {
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 13px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
<div class="chat-header">
    <div style="font-size:32px;">🎓</div>
    <div>
        <h2>MHT-CET Student Helpdesk</h2>
        <p>AI-powered assistant for students &amp; parents &bull; Powered by Claude</p>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# SESSION STATE
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "client" not in st.session_state:
    st.session_state.client = anthropic.Anthropic()

# -------------------------------
# SUGGESTED QUESTIONS
# -------------------------------
SUGGESTIONS = [
    "What is MHT-CET?",
    "Is there negative marking?",
    "How do CAP rounds work?",
    "What is a percentile score?",
    "COEP cutoff marks?",
    "What documents are needed?",
    "How to apply for scholarship?",
    "JEE vs MHT-CET difference?",
]

st.markdown("**Quick questions:**")
cols = st.columns(4)
for i, suggestion in enumerate(SUGGESTIONS):
    if cols[i % 4].button(suggestion, key=f"chip_{i}", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": suggestion})
        st.rerun()

st.divider()

# -------------------------------
# DISPLAY CHAT HISTORY
# -------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-label user">You</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg-label">🤖 MHT-CET Assistant</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-bubble">{msg["content"]}<br><span class="source-badge">MHT-CET AI</span></div>', unsafe_allow_html=True)

# -------------------------------
# AUTO-RESPOND IF LAST MSG IS USER
# -------------------------------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.spinner("Thinking..."):
        try:
            api_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
            response = st.session_state.client.messages.create(
                model="claude-opus-4-5",
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                messages=api_messages
            )
            reply = response.content[0].text
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()
        except Exception as e:
            st.error(f"Error contacting AI: {e}. Please check your ANTHROPIC_API_KEY.")

# -------------------------------
# INPUT BOX
# -------------------------------
st.markdown("<br>", unsafe_allow_html=True)
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Ask your question",
            placeholder="e.g. What is the eligibility for MHT-CET?",
            label_visibility="collapsed"
        )
    with col2:
        submitted = st.form_submit_button("Send", use_container_width=True)

if submitted and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})
    st.rerun()

# -------------------------------
# CLEAR CHAT
# -------------------------------
if st.session_state.messages:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear conversation", use_container_width=False):
        st.session_state.messages = []
        st.rerun()

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<hr style="margin-top:30px; border-color:#e2e8f0;">
<p style="text-align:center; font-size:12px; color:#94a3b8;">
    For official information visit 
    <a href="https://cetcell.mahacet.org" target="_blank">cetcell.mahacet.org</a>
    &bull; This chatbot provides general guidance only
</p>
""", unsafe_allow_html=True)
