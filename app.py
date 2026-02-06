import streamlit as st
import pandas as pd
import pickle
import requests
from datetime import datetime
import time

st.set_page_config(page_title="Healthcare Fraud Detection", layout="wide", page_icon="🏥")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

def query_llm(prompt: str) -> str:
    API_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer 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",
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": "meta-llama/Llama-3.1-8B-Instruct:novita",

    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "⚠️ LLM request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"⚠️ LLM API error: {str(e)}"
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"

model = load_model()

FRAUD_LABELS = ['Phantom Billing', 'Wrong Diagnosis', 'Ghost Enrollee', 'No Fraud']

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        animation: fadeIn 1s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        animation: slideIn 0.5s;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    .no-fraud {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .phantom-billing {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white;
    }
    .wrong-diagnosis {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .ghost-enrollee {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .llm-response {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        animation: fadeIn 0.8s;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    .probability-bar {
        background: #e0e0e0;
        border-radius: 10px;
        height: 30px;
        margin: 5px 0;
        overflow: hidden;
    }
    .probability-fill {
        height: 100%;
        display: flex;
        align-items: center;
        padding-left: 10px;
        color: white;
        font-weight: bold;
        transition: width 0.5s;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏥 Healthcare Fraud Detection System</h1>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Make Prediction", "💬 Interactive Q&A", "📝 Investigation Report", "🎯 Decision Recommendation"])

with tab1:
    st.markdown("### Enter Patient Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        patient_id = st.text_input("Patient ID", value="1")
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        date_encounter = st.date_input("Date of Encounter", value=datetime.now())
    
    with col2:
        date_discharge = st.date_input("Date of Discharge", value=datetime.now())
        amount_billed = st.number_input("Amount Billed", min_value=0.0, value=5000.0, step=100.0)
        diagnosis = st.text_input("Diagnosis", value="Routine Checkup")
    
    if st.button("🔍 Analyze Claim", use_container_width=True):
        with st.spinner("Analyzing claim..."):
            time.sleep(0.5)
            
            input_data = pd.DataFrame({
                'Patient ID': [patient_id],
                'AGE': [age],
                'GENDER': [gender],
                'DATE OF ENCOUNTER': [date_encounter.strftime("%Y-%m-%d")],
                'DATE OF DISCHARGE': [date_discharge.strftime("%Y-%m-%d")],
                'Amount Billed': [amount_billed],
                'DIAGNOSIS': [diagnosis]
            })
            
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            predicted_fraud_type = FRAUD_LABELS[prediction]
            confidence = prediction_proba[prediction] * 100
            
            fraud_class_map = {
                'No Fraud': 'no-fraud',
                'Phantom Billing': 'phantom-billing',
                'Wrong Diagnosis': 'wrong-diagnosis',
                'Ghost Enrollee': 'ghost-enrollee'
            }
            
            fraud_class = fraud_class_map[predicted_fraud_type]
            
            fraud_emoji_map = {
                'No Fraud': '✅',
                'Phantom Billing': '🚨',
                'Wrong Diagnosis': '⚠️',
                'Ghost Enrollee': '👻'
            }
            
            emoji = fraud_emoji_map[predicted_fraud_type]
            
            st.markdown(f"""
                <div class="prediction-card {fraud_class}">
                    <h2>{emoji} Prediction Result</h2>
                    <h3>Fraud Type: {predicted_fraud_type}</h3>
                    <h4>Confidence: {confidence:.2f}%</h4>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Fraud Type Probabilities")
            
            colors = {
                'No Fraud': '#38ef7d',
                'Phantom Billing': '#ff6a00',
                'Wrong Diagnosis': '#f5576c',
                'Ghost Enrollee': '#00f2fe'
            }
            
            for i, label in enumerate(FRAUD_LABELS):
                prob = prediction_proba[i] * 100
                color = colors[label]
                st.markdown(f"""
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: {prob}%; background: {color};">
                            {label}: {prob:.2f}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.session_state['last_prediction'] = {
                'fraud_type': predicted_fraud_type,
                'confidence': confidence,
                'data': {
                    'Patient ID': patient_id,
                    'AGE': age,
                    'GENDER': gender,
                    'DATE OF ENCOUNTER': date_encounter.strftime("%Y-%m-%d"),
                    'DATE OF DISCHARGE': date_discharge.strftime("%Y-%m-%d"),
                    'Amount Billed': amount_billed,
                    'DIAGNOSIS': diagnosis
                },
                'probabilities': {FRAUD_LABELS[i]: prediction_proba[i] * 100 for i in range(len(FRAUD_LABELS))}
            }
            
            with st.spinner("Generating AI explanation..."):
                fraud_definitions = {
                    'Phantom Billing': 'billing for services or procedures that were never actually performed',
                    'Wrong Diagnosis': 'deliberately providing incorrect diagnosis to justify unnecessary procedures or higher billing',
                    'Ghost Enrollee': 'billing for patients who do not exist or are not actually receiving care',
                    'No Fraud': 'legitimate claim with no fraudulent activity detected'
                }
                
                prompt = f"""You are a healthcare fraud detection expert. Explain in simple, non-technical language why this claim was classified as {predicted_fraud_type}.

{predicted_fraud_type} means: {fraud_definitions[predicted_fraud_type]}

Claim Details:
- Patient ID: {patient_id}
- Age: {age}
- Gender: {gender}
- Amount Billed: ${amount_billed}
- Diagnosis: {diagnosis}
- Date of Encounter: {date_encounter}
- Date of Discharge: {date_discharge}

Model Confidence: {confidence:.2f}%

Other Probabilities:
{', '.join([f"{k}: {v:.1f}%" for k, v in st.session_state['last_prediction']['probabilities'].items()])}

Provide a clear, concise explanation that a non-technical person can understand. Focus on the key factors that influenced this classification."""
                
                explanation = query_llm(prompt)
                
                st.markdown(f"""
                    <div class="llm-response">
                        <h4>🤖 AI Explanation</h4>
                        <p>{explanation}</p>
                    </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 💬 Ask Questions About Your Claim")
    
    if 'last_prediction' not in st.session_state:
        st.info("⚠️ Please make a prediction in the first tab before using Q&A.")
    else:
        user_question = st.text_input("Ask a question about the claim:", placeholder="e.g., Why was this claim flagged as Phantom Billing?")
        
        if st.button("Ask AI", use_container_width=True):
            if user_question:
                with st.spinner("AI is thinking..."):
                    pred_data = st.session_state['last_prediction']
                    
                    prompt = f"""You are a healthcare fraud detection assistant. Answer the following question about this claim:

Question: {user_question}

Claim Information:
- Fraud Type: {pred_data['fraud_type']}
- Confidence: {pred_data['confidence']:.2f}%
- Patient ID: {pred_data['data']['Patient ID']}
- Age: {pred_data['data']['AGE']}
- Gender: {pred_data['data']['GENDER']}
- Amount Billed: ${pred_data['data']['Amount Billed']}
- Diagnosis: {pred_data['data']['DIAGNOSIS']}
- Encounter Date: {pred_data['data']['DATE OF ENCOUNTER']}
- Discharge Date: {pred_data['data']['DATE OF DISCHARGE']}

Fraud Type Probabilities:
{', '.join([f"{k}: {v:.1f}%" for k, v in pred_data['probabilities'].items()])}

Provide a helpful, accurate answer based on this information."""
                    
                    answer = query_llm(prompt)
                    
                    st.markdown(f"""
                        <div class="llm-response">
                            <h4>❓ Your Question</h4>
                            <p><i>{user_question}</i></p>
                            <h4>✅ AI Answer</h4>
                            <p>{answer}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a question.")

with tab3:
    st.markdown("### 📝 Auto-Generated Investigation Report")
    
    if 'last_prediction' not in st.session_state:
        st.info("⚠️ Please make a prediction in the first tab before generating a report.")
    else:
        if st.button("📄 Generate Investigation Report", use_container_width=True):
            with st.spinner("Generating comprehensive report..."):
                pred_data = st.session_state['last_prediction']
                
                prompt = f"""Generate a structured investigation report for this healthcare claim:

**Claim Summary:**
- Patient ID: {pred_data['data']['Patient ID']}
- Age: {pred_data['data']['AGE']}
- Gender: {pred_data['data']['GENDER']}
- Amount Billed: ${pred_data['data']['Amount Billed']}
- Diagnosis: {pred_data['data']['DIAGNOSIS']}
- Encounter Date: {pred_data['data']['DATE OF ENCOUNTER']}
- Discharge Date: {pred_data['data']['DATE OF DISCHARGE']}

**Prediction:**
- Fraud Type: {pred_data['fraud_type']}
- Confidence: {pred_data['confidence']:.2f}%

**All Fraud Type Probabilities:**
{', '.join([f"{k}: {v:.1f}%" for k, v in pred_data['probabilities'].items()])}

Generate a professional investigation report with these sections:
1. Executive Summary
2. Fraud Classification Details
3. Risk Factors Identified
4. Suspicious Patterns (if any)
5. Recommendations for Investigation Team

Keep it concise, professional, and actionable."""
                
                report = query_llm(prompt)
                
                st.markdown(f"""
                    <div class="llm-response">
                        <h3>📋 Investigation Report</h3>
                        <p>{report}</p>
                    </div>
                """, unsafe_allow_html=True)

with tab4:
    st.markdown("### 🎯 AI-Powered Decision Recommendation")
    
    if 'last_prediction' not in st.session_state:
        st.info("⚠️ Please make a prediction in the first tab before getting recommendations.")
    else:
        if st.button("🚀 Get Decision Recommendation", use_container_width=True):
            with st.spinner("Analyzing and generating recommendations..."):
                pred_data = st.session_state['last_prediction']
                
                prompt = f"""Based on this healthcare claim fraud analysis, provide specific decision recommendations:

**Claim Details:**
- Fraud Classification: {pred_data['fraud_type']}
- Model Confidence: {pred_data['confidence']:.2f}%
- Amount: ${pred_data['data']['Amount Billed']}
- Diagnosis: {pred_data['data']['DIAGNOSIS']}
- Patient ID: {pred_data['data']['Patient ID']}

**Fraud Type Breakdown:**
{', '.join([f"{k}: {v:.1f}%" for k, v in pred_data['probabilities'].items()])}

Based on the fraud type ({pred_data['fraud_type']}), provide clear recommendations on:

1. **Immediate Action**: 
   - For "No Fraud": Should this claim be auto-approved?
   - For "Phantom Billing": What urgent verification is needed?
   - For "Wrong Diagnosis": What medical records should be reviewed?
   - For "Ghost Enrollee": What identity verification steps are required?

2. **Priority Level**: How urgent is the review? (High/Medium/Low)

3. **Verification Steps**: What specific documents or information should be verified?

4. **Investigation Team Assignment**: Which department should handle this?

5. **Timeline**: Suggested timeframe for action

Be specific and actionable based on the fraud type."""
                
                recommendation = query_llm(prompt)
                
                st.markdown(f"""
                    <div class="llm-response">
                        <h3>🎯 Decision Recommendation</h3>
                        <p>{recommendation}</p>
                    </div>
                """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📊 System Statistics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Predictions", 1 if 'last_prediction' in st.session_state else 0)
with col2:
    st.metric("AI Model", "Llama 3.1 8B")
with col3:
    st.metric("Fraud Types", len(FRAUD_LABELS))
with col4:
    st.metric("Status", "🟢 Online")