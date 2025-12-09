import streamlit as st
import zipfile
import pandas as pd
import pickle
import plotly.express as px

# Load models
reg_model = pickle.load(open("app/models/regression_pipeline.pkl", "rb"))
cls_model = pickle.load(open("app/models/classification_pipeline.pkl", "rb"))

# Load dataset for insights dashboard
# Load cleaned_data.zip
zip_path = "data/processed/cleaned_data.zip"

with zipfile.ZipFile(zip_path, "r") as z:
    with z.open("cleaned_data.csv") as f:
        df = pd.read_csv(f)

# --------------------------------------
# Page Setup
# --------------------------------------
st.set_page_config(page_title="Real Estate Advisor", layout="wide")

# Custom CSS
st.markdown("""
<style>
body { background: #f5f7fa; }
.header {
    padding: 25px;
    color: white;
    text-align: center;
    border-radius: 12px;
    background: linear-gradient(90deg, #4C7EFF, #6A5BFF);
    margin-bottom: 20px;
}
.card {
    background: rgba(255,255,255,0.8);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.result-good {
    background: #e1f7e7; padding: 20px;
    border-left: 6px solid #20c951; border-radius: 12px;
}
.result-bad {
    background: #ffe5e5; padding: 20px;
    border-left: 6px solid #ff4b4b; border-radius: 12px;
}
.result-price {
    background: #eef3ff; padding: 20px;
    border-left: 6px solid #4C7EFF; border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------
st.sidebar.title("🧭 Navigation")
menu = st.sidebar.radio("Go to:", ["🏡 Investment Predictor", "📊 Insights Dashboard"])

# =========================================================
# 1️⃣ PAGE 1 — INVESTMENT PREDICTOR
# =========================================================
if menu == "🏡 Investment Predictor":

    st.markdown("<div class='header'><h1>🏡 Real Estate Investment Advisor</h1><p>AI-powered property investment prediction</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📋 Property Details")

    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("City", sorted(df["City"].unique()))
        property_type = st.selectbox("Property Type", sorted(df["Property_Type"].unique()))
        furnished = st.selectbox("Furnished Status", sorted(df["Furnished_Status"].unique()))
        amenities_count = st.slider("Amenities Count", 1, 5, value=3)

    with col2:
        size_sqft = st.number_input("Size (SqFt)", 200, 10000, step=50)
        price_lakhs = st.number_input("Current Price (Lakhs)", 5.0, 10000.0)
        age = st.number_input("Age of Property (Years)", 0, 50)
        public_transport = st.selectbox("Public Transport Accessibility", sorted(df["Public_Transport_Accessibility"].unique()))
        parking = st.selectbox("Parking Space", ["Yes", "No"])
        security = st.selectbox("Security", ["Yes", "No"])
        facing = st.selectbox("Facing", sorted(df["Facing"].unique()))
        owner_type = st.selectbox("Owner Type", sorted(df["Owner_Type"].unique()))
        availability = st.selectbox("Availability Status", sorted(df["Availability_Status"].unique()))

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔮 Predict Investment & Future Price", use_container_width=True):

        input_df = pd.DataFrame({
            "Size_in_SqFt": [size_sqft],
            "Price_in_Lakhs": [price_lakhs],
            "Amenities_Count": [amenities_count],
            "Nearby_Schools": [1],
            "Nearby_Hospitals": [1],
            "Age_of_Property": [age],
            "City": [city],
            "Property_Type": [property_type],
            "Furnished_Status": [furnished],
            "Public_Transport_Accessibility": [public_transport],
            "Parking_Space": [parking],
            "Security": [security],
            "Facing": [facing],
            "Owner_Type": [owner_type],
            "Availability_Status": [availability]
        })

        pred_class = cls_model.predict(input_df)[0]
        conf = cls_model.predict_proba(input_df)[0].max() * 100
        future_price = reg_model.predict(input_df)[0]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Prediction Results")

        if pred_class == 1:
            st.markdown(f"<div class='result-good'><h3>✔ Good Investment</h3><p>Confidence: <b>{conf:.2f}%</b></p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-bad'><h3>✘ Not a Good Investment</h3><p>Confidence: <b>{conf:.2f}%</b></p></div>", unsafe_allow_html=True)

        st.markdown(f"<div class='result-price'><h3>💰 Estimated Price After 5 Years</h3><p><b>₹ {future_price:.2f} Lakhs</b></p></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)



# =========================================================
# 2️⃣ PAGE 2 — INSIGHTS DASHBOARD
# =========================================================
elif menu == "📊 Insights Dashboard":

    st.markdown("<div class='header'><h1>📊 Insights Dashboard</h1><p>Market insights generated from dataset</p></div>", unsafe_allow_html=True)

    # ---------- Chart 1 ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏙️ Average Current Price by City")
    city_price = df.groupby("City")["Price_in_Lakhs"].mean().reset_index()
    fig1 = px.bar(city_price.head(20), x="City", y="Price_in_Lakhs", color="Price_in_Lakhs",
                  title="Top 20 Cities by Current Price")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Chart 2 ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Top 10 Most Appreciating Cities")
    top_growth = df.groupby("City")["Growth_Rate"].mean().sort_values(ascending=False).head(10).reset_index()
    fig2 = px.bar(top_growth, x="City", y="Growth_Rate", color="Growth_Rate")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Chart 3 ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏢 Property Type vs Future Price")
    fig3 = px.box(df, x="Property_Type", y="Future_Price_5Y", color="Property_Type")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Chart 4 ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎛️ Amenities Count vs Future Price")
    fig4 = px.scatter(df, x="Amenities_Count", y="Future_Price_5Y",
                      color="City", opacity=0.6)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)