import streamlit as st
import requests

st.title("📉 Churn Prediction Uygulaması")

st.markdown("Lütfen müşteri bilgilerini girin:")

features_map = {
    "Evli misiniz?": {"Hayır": 0, "Evet": 1},
    "Telefon hizmeti": {"Hayır": 0, "Evet": 1},
    "Birden fazla hat": {"Hayır": 0, "Evet": 1},
    "İnternet hizmeti": {"Yok": 0, "DSL": 1, "Fiber Optik": 2},
    "Çevrimiçi güvenlik": {"Hayır": 0, "Evet": 1},
    "Çevrimiçi yedekleme": {"Hayır": 0, "Evet": 1},
    "Cihaz koruma": {"Hayır": 0, "Evet": 1},
    "Teknik destek": {"Hayır": 0, "Evet": 1},
    "TV yayını": {"Hayır": 0, "Evet": 1},
    "Film yayını": {"Hayır": 0, "Evet": 1},
    "Sözleşme türü": {"Aylık": 0, "1 Yıllık": 1, "2 Yıllık": 2},
    "Kağıtsız fatura": {"Hayır": 0, "Evet": 1},
    "Ödeme yöntemi": {
        "Elektronik çek": 0,
        "Otomatik banka ödemesi": 1,
        "Kredi kartı": 2,
        "Mektupla çek": 3
    }
}

user_inputs = {}

for label, options in features_map.items():
    selected_label = st.radio(label, options=list(options.keys()))
    user_inputs[label] = options[selected_label]

# Sayısal girişler
user_inputs["Toplam ödeme"] = st.number_input("Toplam ödeme", min_value=0.0)

if 'gecmis' not in st.session_state:
    st.session_state.gecmis=[]


if st.button("🔍 Tahmin Et"):
    veri = user_inputs
    try:
        response = requests.post('https://churn-prediction-5f8q.onrender.com/predict', json=veri)
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
        st.json(log['Girdi'])
        
        














