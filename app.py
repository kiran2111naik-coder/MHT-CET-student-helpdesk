import streamlit as st
import nltk
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download punkt
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"NLTK download issue: {e}")

# -------------------------------
# FAQ DATA (MHT-CET)
# -------------------------------
faq_questions = [
    "What is MHT CET?",
    "Who conducts MHT CET exam?",
    "What is eligibility for MHT CET?",
    "What is exam pattern of MHT CET?",
    "Is there negative marking?",
    "How to apply for MHT CET?",
    "What is syllabus for MHT CET?",
    "When is MHT CET exam conducted?",
    "What is cutoff for engineering?",
    "How to prepare for MHT CET?",
    "Can parents track admission process?",
    "What documents are required for admission?",
    "How CAP rounds work?",
    "What is best college through MHT CET?",
    "Is MHT CET easier than JEE?",

    "What is MHT CET application fee?",
    "Can I edit my application form?",
    "What is PCM and PCB group?",
    "Can I give both PCM and PCB exams?",
    "How many attempts are allowed for MHT CET?",
    "Is Aadhaar card mandatory?",
    "How to download admit card?",
    "What to carry in exam hall?",
    "What is exam duration?",
    "How marks are calculated in MHT CET?",
    "What is percentile score?",
    "Is there normalization in MHT CET?",
    "How to check MHT CET result?",
    "What after MHT CET result?",
    "How to register for CAP rounds?",
    "How many CAP rounds are there?",
    "Can I change college preference in CAP?",
    "What is seat allotment?",
    "What if I don't get seat in CAP?",
    "What is spot round admission?",
    "Can parents attend counselling?",
    "What is minority quota?",
    "What is TFWS seat?",
    "What is domicile certificate?",
    "Is gap certificate required?",
    "What is caste validity certificate?",
    "How to choose best college?",
    "What is difference between autonomous and non autonomous college?",
    "What branches are available in engineering?",
    "Which branch is best for future?",
    "What is fees for engineering colleges?",
    "Are scholarships available?",
    "How to apply for scholarship?",
    "Can I get hostel facility?",
    "Is laptop required for engineering?",
    "How to prepare in last month?",
    "Best books for MHT CET preparation?",
    "How many questions in exam?",
    "Is calculator allowed?",
    "Can I change exam center?",
    "What happens if I miss exam?",
    "Is re exam possible?",
    "Can I take drop for MHT CET?",
    "Is coaching necessary for MHT CET?",
    "How many marks required for COEP?",
    "What is VJTI cutoff?",
    "Difference between JEE and MHT CET?",
    "Can I get admission without MHT CET?",
    "What is lateral entry admission?"
]

faq_answers = [
    "MHT CET is an entrance exam for engineering and pharmacy colleges in Maharashtra.",
    "MHT CET is conducted by State CET Cell, Maharashtra.",
    "Students must have passed 12th with Physics, Chemistry and Maths/Biology.",
    "The exam is online and includes Physics, Chemistry, Maths/Biology questions.",
    "No, there is no negative marking in MHT CET.",
    "You can apply online through the official CET website.",
    "Syllabus is based on Maharashtra State Board 11th and 12th.",
    "Usually conducted between April and May.",
    "Cutoff varies every year depending on college and branch.",
    "Practice MCQs, revise concepts, and solve previous year papers.",
    "Yes, parents can check updates through official CAP portal.",
    "Documents include marksheets, CET scorecard, ID proof etc.",
    "CAP rounds are centralized admission rounds based on merit.",
    "Top colleges include COEP, VJTI, SPIT.",
    "Yes, generally it is easier compared to JEE Main.",

    "Application fee is around 800 to 1000 rupees depending on category.",
    "Yes, correction window is provided for limited time.",
    "PCM is Physics Chemistry Maths, PCB is Physics Chemistry Biology.",
    "Yes, you can appear for both groups.",
    "There is no attempt limit.",
    "Yes, Aadhaar is generally required for identification.",
    "Download admit card from official website using login.",
    "Carry admit card and valid ID proof.",
    "Exam duration is around 3 hours.",
    "Marks are based on correct answers, no negative marking.",
    "Percentile shows your rank compared to other students.",
    "Yes, normalization may be applied.",
    "You can check result online using login credentials.",
    "You need to participate in CAP rounds for admission.",
    "Register online on CET CAP portal.",
    "Usually 3 CAP rounds are conducted.",
    "Yes, you can modify preferences in certain rounds.",
    "Seat allotment is based on rank and preferences.",
    "You can try next rounds or spot round.",
    "Spot rounds are conducted by colleges for vacant seats.",
    "Yes, parents can attend counselling process.",
    "Minority quota is reserved for specific communities.",
    "TFWS seats provide tuition fee waiver.",
    "Domicile proves you are Maharashtra resident.",
    "Gap certificate needed if there is study gap.",
    "Required for reserved category students.",
    "Consider placement, faculty and location.",
    "Autonomous colleges have flexible syllabus.",
    "Branches include CS, IT, Mechanical, Civil etc.",
    "CS and IT are popular currently.",
    "Fees range from 50k to 2 lakhs per year.",
    "Yes, government scholarships are available.",
    "Apply through MahaDBT portal.",
    "Many colleges provide hostel facility.",
    "Yes, laptop is useful for studies.",
    "Revise important topics and solve mock tests.",
    "Refer state board books and MCQ practice books.",
    "Around 150 questions are asked.",
    "No, calculator is not allowed.",
    "Limited option to change center before exam.",
    "You cannot give exam again that year.",
    "No re exam usually.",
    "Yes, you can take drop and reappear.",
    "Not necessary but helpful.",
    "Above 99 percentile required for COEP.",
    "VJTI cutoff is also very high.",
    "JEE is national level, CET is state level.",
    "Yes, through management quota.",
    "Lateral entry is direct 2nd year admission for diploma students."
]

# -------------------------------
# NLP MODEL
# -------------------------------
SIMILARITY_THRESHOLD = 0.4  # Increased from 0.3 for better accuracy

@st.cache_resource
def initialize_vectorizer():
    """Initialize and cache TF-IDF vectorizer to avoid recomputation."""
    try:
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        X = vectorizer.fit_transform(faq_questions)
        return vectorizer, X
    except Exception as e:
        logger.error(f"Vectorizer initialization error: {e}")
        st.error("Error initializing chatbot. Please refresh.")
        return None, None

vectorizer, X = initialize_vectorizer()

def get_response(user_input: str) -> dict:
    """Get response with confidence score.
    
    Args:
        user_input: User's question
        
    Returns:
        dict with 'answer', 'confidence', and 'success' keys
    """
    if not user_input or not user_input.strip():
        return {
            'answer': "Please enter a question.",
            'confidence': 0.0,
            'success': False
        }
    
    try:
        user_vec = vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vec, X)
        index = similarity.argmax()
        confidence = float(similarity[0][index])
        
        if confidence < SIMILARITY_THRESHOLD:
            logger.info(f"Low confidence match: {confidence:.2f} for '{user_input}'")
            return {
                'answer': "Sorry, I couldn't understand. Please ask questions related to MHT-CET (exam pattern, eligibility, application, colleges, etc.).",
                'confidence': confidence,
                'success': False
            }
        
        logger.info(f"Match found: {confidence:.2f}")
        return {
            'answer': faq_answers[index],
            'confidence': confidence,
            'success': True
        }
    
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return {
            'answer': "An error occurred. Please try again.",
            'confidence': 0.0,
            'success': False
        }

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(
    page_title="MHT-CET Helpdesk Chatbot",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 MHT-CET Student Helpdesk Chatbot")
st.write("Ask your questions about MHT-CET exam, admission, colleges, and preparation (For Students & Parents)")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if vectorizer is None or X is None:
    st.error("Chatbot initialization failed. Please refresh the page.")
    st.stop()

# Sidebar info
with st.sidebar:
    st.subheader("📋 Topics Covered")
    st.write("""
    • Exam Pattern & Eligibility
    • Application & Admission
    • CAP Rounds & Seat Allotment
    • Colleges & Branches
    • Preparation Tips
    • Documents Required
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# User input section
st.subheader("Ask Your Question")
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.text_input(
        "Type your question here...",
        placeholder="e.g., What is eligibility for MHT CET?",
        label_visibility="collapsed"
    )

with col2:
    ask_button = st.button("Ask", use_container_width=True)

if ask_button:
    if user_input:
        result = get_response(user_input)
        
        st.session_state.messages.append(("You", user_input))
        st.session_state.messages.append(("Bot", result['answer'], result['confidence']))
        st.rerun()
    else:
        st.warning("Please enter a question.")

# Display chat history
if st.session_state.messages:
    st.subheader("Chat History")
    for i, msg in enumerate(st.session_state.messages):
        if msg[0] == "You":
            st.write(f"🧑‍🎓 **You:** {msg[1]}")
        else:
            confidence = msg[2] if len(msg) > 2 else 0
            # Show confidence indicator
            confidence_text = f"(Confidence: {confidence:.0%})" if confidence > 0 else ""
            st.write(f"🤖 **Bot:** {msg[1]} {confidence_text}")
            
            # Visual confidence indicator
            if confidence > 0:
                st.progress(confidence, text=f"Match accuracy: {confidence:.1%}")
