import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Breast Cancer Prediction AI",
    page_icon="aiBrain.png",
    layout="wide"
)


st.title("🩺 نظام الذكاء الاصطناعي للتنبؤ واكتشاف سرطان الثدي")
st.markdown("""
هذا التطبيق يستخدم نموذج تعلم آلة (Random Forest) تم تدريبه على بيانات Wisconsin الطبية العالمية 
لمساعدة الأطباء في التنبؤ بطبيعة الأورام بناءً على القياسات المستخرجة من الفحص المجهري للخلية.
""")
st.write("---")

# 💾 2. تحميل الموديل بأمان باستخدام التدابير الاحتياطية
@st.cache_resource  
def load_my_model():
    try:
      
        model = joblib.load("MLBREATCANCER/breastCancerPrediction.joblib")
        return model
    except FileNotFoundError:
        st.error("❌يتعذر تحميل الذكاء الاصطناعي ")
       
        return None

model = load_my_model()


if model is not None:
    
    st.sidebar.header("📥 إدخال القياسات الطبية للحالة")
    st.sidebar.markdown("قومي بتعديل المؤشرات بناءً على تقرير التحليل المخبري:")

    
    mean_radius = st.sidebar.slider("متوسط نصف قطر الخلية (Mean Radius)", 5.0, 30.0, 14.0, step=0.1)
    mean_texture = st.sidebar.slider("نسيج الخلية (Mean Texture)", 5.0, 40.0, 19.0, step=0.1)
    mean_perimeter = st.sidebar.slider("محيط الخلية (Mean Perimeter)", 40.0, 190.0, 90.0, step=0.1)
    mean_area = st.sidebar.slider("مساحة الخلية (Mean Area)", 100.0, 2500.0, 650.0, step=1.0)
    mean_smoothness = st.sidebar.slider("نعومة الخلية (Mean Smoothness)", 0.05, 0.25, 0.10, step=0.01)
    
      
    input_features = np.zeros(30)
    input_features[0] = mean_radius
    input_features[1] = mean_texture
    input_features[2] = mean_perimeter
    input_features[3] = mean_area
    input_features[4] = mean_smoothness
   
    input_features[5:] = [0.1, 0.1, 0.05, 0.15, 0.05, 0.3, 1.0, 2.0, 20.0, 0.005, 0.02, 0.02, 0.01, 0.02, 0.003, 16.0, 25.0, 100.0, 800.0, 0.12, 0.25, 0.3, 0.11, 0.29, 0.08]

    
    final_features = np.array([input_features])

    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 ملخص المؤشرات الحيوية الحالية")
        current_data = pd.DataFrame({
            "الميزة الطبية": ["Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness"],
            "القيمة الحالية": [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]
        })
        st.table(current_data)


    with col2:
        st.subheader("🎯 تحليل الذكاء الاصطناعي")
        
        
        if st.button("تشغيل فحص الموديل الفوري", type="primary"):
            
            
            prediction = model.predict(final_features)
            
            
            try:
                prediction_proba = model.predict_proba(final_features)[0]
                malignant_prob = prediction_proba[0] * 100
                benign_prob = prediction_proba[1] * 100
            except:
                malignant_prob = 50.0 if prediction[0] == 0 else 0.0
                benign_prob = 100.0 - malignant_prob

            st.write("---")
            #: 0 => (Malignant) but 1=> (Benign)
            if prediction[0] == 0:
                st.error(f"🚨 النتيجة المتوقعة: ورم خبيث (Malignant) ")
                st.warning(f"📊 درجة اليقين الإحصائي للإصابة: {malignant_prob:.2f}%")
                
                st.progress(int(malignant_prob))
            else:
                st.success(f"🟢 النتيجة المتوقعة: ورم حميد وآمن (Benign) 😊")
                st.info(f"📊 درجة يقين الموديل بسلامة الحالة: {benign_prob:.2f}%")
                
                st.progress(int(benign_prob))

st.write("---")
st.caption("All Right Reseved to Ls")
