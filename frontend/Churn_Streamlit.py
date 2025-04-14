import streamlit as st
import requests

st.title("📉 Churn Prediction Uygulaması")

st.markdown("Bu uygulama, müşteri verilerine göre abonelik iptali riskini tahmin eder.")

features_map = {
    "SeniorCitizen": {"Hayır": 0, "Evet": 1},
    "Partner": {"Hayır": 0, "Evet": 1},
    "Dependents": {"Hayır": 0, "Evet": 1},
    "MultipleLines": {"Hayır": 0, "Evet": 1},
    "InternetService": {"Yok": 0, "DSL": 1, "Fiber Optik": 2},
    "OnlineSecurity": {"Hayır": 0, "Evet": 1},
    "OnlineBackup": {"Hayır": 0, "Evet": 1},
    "DeviceProtection": {"Hayır": 0, "Evet": 1},
    "TechSupport": {"Hayır": 0, "Evet": 1},
    "StreamingTV": {"Hayır": 0, "Evet": 1},
    "StreamingMovies": {"Hayır": 0, "Evet": 1},
    "Contract": {"Aylık": 0, "1 Yıllık": 1, "2 Yıllık": 2},
    "PaperlessBilling": {"Hayır": 0, "Evet": 1},
    "PaymentMethod": {
        "Elektronik çek": 0,
        "Otomatik banka ödemesi": 1,
        "Kredi kartı": 2,
        "Mektupla çek": 3
    }
}


user_inputs = {}
for key, label_options in features_map.items():
    label = st.radio(f"{key} seçin", options=list(label_options.keys()))
    user_inputs[key] = label_options[label]


# Sayısal girişler
user_inputs["MonthlyCharges"] = st.number_input("Aylık ödeme")

if 'gecmis' not in st.session_state:
    st.session_state.gecmis=[]


if st.button("🔍 Tahmin Et"):
    veri = user_inputs
    try:
        response = requests.post('https://churn-prediction-5f8q.onrender.com/predict', json={'data':veri})
        result = response.json()
        st.success(f"📌 Churn Tahmini: {'Evet (1)' if result['churn']==1 else 'Hayır (0)'}")
        st.info(f"🎯 Churn Olasılığı: %{result['probability']*100:.2f}")
        st.session_state.gecmis.append({
            "Data": veri,
            "Churn": result['churn'],
            "Probability": f"%{result['probability']*100:.2f}"
        })
    except Exception as e:
        st.error(f"❌ Tahmin sırasında hata: {e}")


if st.session_state.gecmis:
    st.subheader("Tahmin Gecmisi")
    for i, log in enumerate(reversed(st.session_state.gecmis), start=1):
        st.write(f"Churn: {log['Churn']}, Olasilik: {log['Probability']}")
        st.json(log['Data'])

        














