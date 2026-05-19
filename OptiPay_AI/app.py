from pyexpat import model
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

model = joblib.load("best_model.pkl")
le_dict = joblib.load("label_encoders.pkl")
# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="OptiPay AI", layout="centered")

# -------------------------------
# CSS (UPDATED UI)
# -------------------------------
st.markdown("""
<style>
.main { background-color: #0E1117; }

h1, h2, h3 {
    text-align: center;
}

/* Button */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
    background-color: #FF4B4B;
    color: white;
}
.stButton>button:hover {
    background-color: #ff7a7a;
    color: #800000;
}

/* Result box */
.success-box {
    background-color: #1f4037;
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
}

/* Explanation box */
.explain-box {
    background-color: #1E2A38;
    padding: 15px;
    border-radius: 10px;
    color: white;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.title("OptiPay AI")
st.subheader("Smart Payment Recommendation System")

# -------------------------------
# CATEGORY MAP
# -------------------------------
category_platform_map = {
    "Fashion": ["Flipkart", "Amazon", "Myntra", "Ajio", "Puma", "Nike", "H&M"],
    "Electronics": ["Vijay Sales", "Croma", "Reliance Digital"],
    "Food Delivery": ["Zomato", "Swiggy", "EatClub", "Box8"],
    "Travel": ["MakeMyTrip", "RedBus", "Goibibo", "IRCTC"],
    "Grocery": ["Zepto", "Blinkit", "Instamart", "Flipkart Minutes", "BigBasket", "Amazon Fresh"]
}

# -------------------------------
# INPUTS
# -------------------------------
amount = st.number_input("Enter Transaction Amount (₹)", min_value=0, step=100, value=1000)

category = st.selectbox("Choose Category", ["Select Category"] + list(category_platform_map.keys()))

platform = st.selectbox(
    "Choose Platform",
    ["Select Platform"] + category_platform_map.get(category, [])
)

# -------------------------------
# FUNCTIONS (UNCHANGED)
# -------------------------------
def get_offers(category, platform):
        # -------------------------------
    # 🔹 FASHION
    # -------------------------------
    if category == "Fashion":

        if platform == "Myntra":
            return [
                {"bank": "HDFC", "method": "Credit Card", "percent": 15, "max": 1200, "min_txn": 3000},
                {"bank": "Axis", "method": "Credit Card", "percent": 12, "max": 900, "min_txn": 2500},
            ]

        elif platform == "Flipkart":
            return [
                {"bank": "SBI", "method": "Credit Card", "percent": 10, "max": 1000, "min_txn": 3000},
                {"bank": "ICICI", "method": "Debit Card", "percent": 8, "max": 700, "min_txn": 2000},
            ]

        elif platform == "Amazon":
            return [
                {"bank": "ICICI", "method": "Credit Card", "percent": 10, "max": 800, "min_txn": 2500},
                {"bank": "HDFC", "method": "Debit Card", "percent": 7, "max": 600, "min_txn": 2000},
            ]

        elif platform == "Ajio":
            return [
                {"bank": "HDFC", "method": "Credit Card", "percent": 14, "max": 900, "min_txn": 2500},
                {"bank": "Axis", "method": "Debit Card", "percent": 10, "max": 700, "min_txn": 2000},
            ]

        elif platform == "Puma":
            return [
                {"bank": "Axis", "method": "Credit Card", "percent": 18, "max": 1000, "min_txn": 3000},
                {"bank": "ICICI", "method": "UPI", "percent": 10, "max": 500, "min_txn": 1500},
            ]

        elif platform == "Nike":
            return [
                {"bank": "HDFC", "method": "Credit Card", "percent": 16, "max": 1100, "min_txn": 3000},
                {"bank": "SBI", "method": "Debit Card", "percent": 9, "max": 600, "min_txn": 2000},
            ]

        elif platform == "H&M":
            return [
                {"bank": "ICICI", "method": "Credit Card", "percent": 13, "max": 900, "min_txn": 2500},
                {"bank": "Axis", "method": "UPI", "percent": 8, "max": 400, "min_txn": 1500},
            ]

    # -------------------------------
    # 🔹 ELECTRONICS
    # -------------------------------
    elif category == "Electronics":

        if platform == "Vijay Sales":
            return [
                {"bank": "HDFC", "method": "Credit Card", "percent": 12, "max": 1500, "min_txn": 10000},
                {"bank": "Axis", "method": "Card", "percent": 10, "max": 1200, "min_txn": 8000},
            ]

        elif platform == "Croma":
            return [
                {"bank": "ICICI", "method": "Credit Card", "percent": 10, "max": 1200, "min_txn": 9000},
                {"bank": "SBI", "method": "Card", "percent": 8, "max": 1000, "min_txn": 7000},
            ]

        elif platform == "Reliance Digital":
            return [
                {"bank": "Axis", "method": "Credit Card", "percent": 11, "max": 1300, "min_txn": 9000},
                {"bank": "HDFC", "method": "Card", "percent": 9, "max": 1100, "min_txn": 8000},
            ]

    # -------------------------------
    # 🔹 FOOD DELIVERY
    # -------------------------------
    elif category == "Food Delivery":

        if platform == "Swiggy":
            return [
                {"bank": "HDFC", "method": "UPI", "percent": 20, "max": 200, "min_txn": 500},
                {"bank": "ICICI", "method": "Wallet", "percent": 12, "max": 150, "min_txn": 400},
            ]

        elif platform == "Zomato":
            return [
                {"bank": "Axis", "method": "UPI", "percent": 15, "max": 120, "min_txn": 300},
                {"bank": "SBI", "method": "Card", "percent": 10, "max": 100, "min_txn": 250},
            ]

        elif platform == "EatClub":
            return [
                {"bank": "ICICI", "method": "UPI", "percent": 18, "max": 180, "min_txn": 400},
                {"bank": "HDFC", "method": "Wallet", "percent": 10, "max": 120, "min_txn": 300},
            ]

        elif platform == "Box8":
            return [
                {"bank": "SBI", "method": "UPI", "percent": 12, "max": 130, "min_txn": 350},
                {"bank": "Axis", "method": "Wallet", "percent": 10, "max": 100, "min_txn": 300},
            ]

    # -------------------------------
    # 🔹 TRAVEL
    # -------------------------------
    elif category == "Travel":

        if platform == "MakeMyTrip":
            return [
                {"bank": "Axis", "method": "Credit Card", "percent": 20, "max": 2000, "min_txn": 15000},
                {"bank": "HDFC", "method": "Card", "percent": 15, "max": 1800, "min_txn": 12000},
            ]

        elif platform == "RedBus":
            return [
                {"bank": "SBI", "method": "UPI", "percent": 12, "max": 600, "min_txn": 2000},
                {"bank": "ICICI", "method": "Card", "percent": 10, "max": 500, "min_txn": 1500},
            ]

        elif platform == "Goibibo":
            return [
                {"bank": "HDFC", "method": "Credit Card", "percent": 18, "max": 1500, "min_txn": 10000},
                {"bank": "Axis", "method": "UPI", "percent": 12, "max": 800, "min_txn": 5000},
            ]

        elif platform == "IRCTC":
            return [
                {"bank": "SBI", "method": "UPI", "percent": 10, "max": 500, "min_txn": 2000},
                {"bank": "ICICI", "method": "Card", "percent": 8, "max": 400, "min_txn": 1500},
            ]

    # -------------------------------
    # 🔹 GROCERY
    # -------------------------------
    elif category == "Grocery":

        if platform == "Instamart":
            return [
                {"bank": "HDFC", "method": "UPI", "percent": 12, "max": 200, "min_txn": 500},
                {"bank": "ICICI", "method": "Wallet", "percent": 10, "max": 150, "min_txn": 400},
            ]

        elif platform == "Flipkart Minutes":
            return [
                {"bank": "Axis", "method": "UPI", "percent": 10, "max": 180, "min_txn": 400},
                {"bank": "SBI", "method": "Wallet", "percent": 8, "max": 120, "min_txn": 300},
            ]

        elif platform == "BigBasket":
            return [
                {"bank": "ICICI", "method": "Card", "percent": 9, "max": 200, "min_txn": 600},
                {"bank": "HDFC", "method": "UPI", "percent": 11, "max": 220, "min_txn": 700},
            ]

        elif platform == "Amazon Fresh":
            return [
                {"bank": "HDFC", "method": "Card", "percent": 10, "max": 250, "min_txn": 800},
                {"bank": "Axis", "method": "UPI", "percent": 8, "max": 180, "min_txn": 600},
            ]

        elif platform == "Blinkit":
            return [
                {"bank": "Axis", "method": "UPI", "percent": 10, "max": 180, "min_txn": 400},
                {"bank": "SBI", "method": "Wallet", "percent": 8, "max": 120, "min_txn": 300},
            ]

        elif platform == "Zepto":
            return [
                {"bank": "HDFC", "method": "UPI", "percent": 12, "max": 200, "min_txn": 500},
                {"bank": "ICICI", "method": "Wallet", "percent": 10, "max": 150, "min_txn": 400},
            ]

    return []

def prepare_input(amount, category, platform):
    return pd.DataFrame([{
        "transaction_amount_inr": amount,
        "category": category,
        "platform": platform
    }])

def calculate_cashback(amount, offer):
    if amount < offer["min_txn"]:
        return 0
    return min((offer["percent"]/100)*amount, offer["max"])

# -------------------------------
# BUTTON (CENTERED)
# -------------------------------
col1, col2, col3 = st.columns([1,2,1])

with col2:
    run = st.button("🚀 Get Best Payment Option")

# -------------------------------
# LOGIC
# -------------------------------
if run:

    if category == "Select Category":
        st.warning("⚠️ Please select category")
        st.stop()

    if platform == "Select Platform":
        st.warning("⚠️ Please select platform")
        st.stop()

    offers = get_offers(category, platform)

    results = []

    for offer in offers:
        try:
            input_df = pd.DataFrame([{
                "user_spending_profile": 0,
                "user_bank_preference": 0,

                "platform": le_dict["platform"].transform(
                    [platform if platform in le_dict["platform"].classes_ else le_dict["platform"].classes_[0]]
                )[0],

                "category": le_dict["category"].transform(
                    [category if category in le_dict["category"].classes_ else le_dict["category"].classes_[0]]
                )[0],

                "transaction_amount_inr": amount,

                "payment_method": le_dict["payment_method"].transform(
                    [offer["method"] if offer["method"] in le_dict["payment_method"].classes_ else le_dict["payment_method"].classes_[0]]
                )[0],

                "bank": le_dict["bank"].transform(
                    [offer["bank"] if offer["bank"] in le_dict["bank"].classes_ else le_dict["bank"].classes_[0]]
                )[0],

                "card_network": 0,

                "cashback_percentage": offer["percent"],
                "max_cashback_inr": offer["max"],
                "min_transaction_inr": offer["min_txn"],

                "offer_applicable": 1,
                "offer_valid_days": 30,

                "potential_cashback": (amount * offer["percent"]) / 100,

                "capped_potential": min(
                    (amount * offer["percent"]) / 100,
                    offer["max"]
                ),

                "txn_above_min": 1 if amount >= offer["min_txn"] else 0,

                "txn_surplus": max(0, amount - offer["min_txn"]),

                "effective_cashback_pct":
                    min((amount * offer["percent"]) / 100, offer["max"]) / amount * 100
            }]
            )

            cb = model.predict(input_df)[0]
        except Exception as e:
            st.error(e)
            cb = 0

        results.append({
            "Bank": offer.get("bank", "N/A"),
            "Method": offer.get("method", "N/A"),
            "Cashback (₹)": round(cb, 2),
            "Savings (%)": round((cb / amount) * 100, 2) if amount > 0 else 0
        })
    df = pd.DataFrame(results)

    if df.empty:
        st.warning("No offers available ❌")
    else:
        df = df.sort_values(by="Cashback (₹)", ascending=False)

        best = df.iloc[0]

        # ✅ BEST OPTION BOX
        st.markdown(f"""
        <div class="success-box">
            <h2>Best Option</h2>
            <h3>{best['Bank']} ({best['Method']})</h3>
            <p>💰 Cashback: ₹{best['Cashback (₹)']}</p>
        </div>
        """, unsafe_allow_html=True)

        # ✅ TOP OFFERS COMPARISON
        st.subheader("Top Offers Comparison")

        # Remove index
        df_display = df.reset_index(drop=True)

        # Convert dataframe to centered HTML table
        table_html = """
        <style>
        .custom-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 18px;
            text-align: center;
            overflow: hidden;
            border-radius: 12px;
        }

        .custom-table th {
            background-color: #1E2A38;
            color: white;
            padding: 14px;
            text-align: center;
        }

        .custom-table td {
            padding: 14px;
            text-align: center;
            border-bottom: 1px solid #ddd;
        }

        .custom-table tr:nth-child(even) {
            background-color: #f5f5f5;
        }
        </style>

        <table class="custom-table">
        <tr>
        """

        # Headers
        for col in df_display.columns:
            table_html += f"<th>{col}</th>"

        table_html += "</tr>"

        # Rows
        for _, row in df_display.iterrows():
            table_html += "<tr>"
            for value in row:
                table_html += f"<td>{value}</td>"
            table_html += "</tr>"

        table_html += "</table>"

        # Show table
        st.markdown(table_html, unsafe_allow_html=True)
        
        best_offer = next(o for o in offers if o["bank"] == best["Bank"])

        # ✅ EXPLANATION BOX
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1E2A38, #243B55);
            padding: 25px;
            border-radius: 15px;
            color: white;
            margin-top: 20px;
            font-family: 'Segoe UI', sans-serif;
        ">
        <h4 style="margin-bottom:10px;">Why this is best?</h4>

        <p style="font-size:15px; line-height:1.8;">
        <b>Cashback Rate:</b> {best_offer['percent']}%<br>
        <b>Max Benefit:</b> ₹{best_offer['max']}<br>
        <b>Minimum Spend:</b> ₹{best_offer['min_txn']}
        </p>

        <hr style="border: 1px solid #444;">

        <p style="font-size:15px;">
        <b>Insight:</b> This option gives you the highest return for your spending, 
        making it the most efficient choice among all available offers.
        </p>

        </div>
        """, unsafe_allow_html=True)