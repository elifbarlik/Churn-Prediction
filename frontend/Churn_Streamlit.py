import streamlit as st
import requests

st.title("📉 Churn Prediction Uygulaması")

st.markdown("Lütfen müşteri bilgilerini girin:")

SeniorCitizen = st.radio("Senior Citizen", [0, 1])
Partner = st.radio("Partner", [0, 1])
Dependents = st.radio("Dependents", [0, 1])
MultipleLines = st.radio("Multiple Lines", [0, 1, 2])
InternetService = st.radio("Internet Service", [0, 1, 2])
OnlineSecurity = st.radio("Online Security", [0, 1, 2])
OnlineBackup = st.radio("Online Backup", [0, 1, 2])
DeviceProtection = st.radio("Device Protection", [0, 1, 2])
TechSupport = st.radio("Tech Support", [0, 1, 2])
StreamingTV = st.radio("Streaming TV", [0, 1, 2])
StreamingMovies = st.radio("Streaming Movies", [0, 1, 2])
Contract = st.radio("Contract", [0, 1, 2])
PaperlessBilling = st.radio("Paperless Billing", [0, 1])
PaymentMethod = st.radio("Payment Method", [0, 1, 2, 3])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)


if 'gecmis' not in st.session_state:
    st.session_state.gecmis=[]


if st.button("🔍 Tahmin Et"):
    veri = {
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges
    }

    try:
        response = requests.post('https://churn-prediction-5f8q.onrender.com/predict', json=veri)
        result = response.json()
        st.success(f"📌 Churn Tahmini: {'Evet (1)' if result['churn']==1 else 'Hayır (0)'}")
        st.info(f"🎯 Churn Olasılığı: %{result['probability']*100:.2f}")
        st.session_state.gecmis.append({
            "Girdi": veri,
            "Churn": result['churn'],
            "Olasilik": f"%{result['probability']*100:.2f}"
        })
    except Exception as e:
        st.error(f"❌ Tahmin sırasında hata: {e}")


if st.session_state.gecmis:
    st.subheader("Tahmin Gecmisi")
    for i, log in enumerate(reversed(st.session_state.gecmis), start=1):
        st.write(f"Churn: {log['Churn']}, Olasilik: {log['Olasilik']}")
        st.json(log['Girdi'])
        
        














