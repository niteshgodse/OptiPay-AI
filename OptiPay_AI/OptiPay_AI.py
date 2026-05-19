#!/usr/bin/env python
# coding: utf-8

# # OptiPay AI: Smart Payment Offer Optimization System
# **Goal:** Predict actual cashback for every available payment offer, rank them, and recommend the best payment method for a given transaction — with a human-readable explanation.
# 
# ---

# ## Step 1: Imports & Data Loading

# In[1]:


import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from scipy.sparse import hstack, csr_matrix

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv('/Users/nitesh/Downloads/optipay_ai_dataset.csv')

print('Shape          :', df.shape)
print('Columns        :', list(df.columns))
print('\nSample rows:')
df.head(3)


# ## Step 2: Data Cleaning

# In[2]:


# ── 2a. Drop ID / date / raw-text columns  (not useful for ML directly) ──────
drop_cols = ['transaction_id', 'user_id', 'offer_start_date', 'offer_end_date']
df.drop(columns=drop_cols, inplace=True)

# ── 2b. Handle missing values ─────────────────────────────────────────────────
print('Nulls before fix:')
print(df.isnull().sum()[df.isnull().sum() > 0])

# card_network is 'N/A' for UPI/Wallet/Net Banking — fill missing with 'N/A'
df['card_network'] = df['card_network'].fillna('N/A')

# ── 2c. Fix boolean column ────────────────────────────────────────────────────
df['offer_applicable'] = df['offer_applicable'].astype(int)   # True/False → 1/0

print('\nNulls after fix:', df.isnull().sum().sum())
print('Shape after cleaning:', df.shape)


# ## Step 3: Encoding Categorical Variables

# In[3]:


# ── Tree-based models (RandomForest / GradientBoosting) do NOT need scaling.
# They split on thresholds, not distances — so raw numeric values work fine.
# We only need to convert string categories to integers.

cat_cols = [
    'user_spending_profile', 'user_bank_preference',
    'platform', 'category', 'payment_method', 'bank', 'card_network'
]

le_dict = {}          # store encoders so we can reuse them at prediction time
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

print('Encoded columns:', cat_cols)
print('\nSample encoded values:')
df[cat_cols].head(3)


# ## Step 4: NLP on Offer Description

# In[4]:


# ── 4a. Text preprocessing ────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s%.]', ' ', text)   # keep alphanumeric, %, .
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_desc'] = df['offer_description'].apply(clean_text)

# ── 4b. TF-IDF Vectorization ──────────────────────────────────────────────────
# max_features=50 keeps the most informative terms (cashback, upi, hdfc, etc.)
# ngram_range=(1,2) captures phrases like '10 cashback', 'credit card'
tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2), stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['clean_desc'])   # sparse matrix (3000 × 50)

print('TF-IDF matrix shape:', tfidf_matrix.shape)
print('\nTop TF-IDF terms:', tfidf.get_feature_names_out()[:20])


# ## Step 5: Feature Engineering

# In[5]:


# ── New derived features (no data leakage — all computed from input fields) ──

# 1. How much cashback could we get without the cap?
df['potential_cashback'] = (df['cashback_percentage'] / 100) * df['transaction_amount_inr']

# 2. After applying the cap — this is what we'd actually earn if offer is applicable
df['capped_potential'] = df[['potential_cashback', 'max_cashback_inr']].min(axis=1)

# 3. Is the transaction amount above the minimum threshold? (binary)
df['txn_above_min'] = (df['transaction_amount_inr'] >= df['min_transaction_inr']).astype(int)

# 4. How far is the transaction from the minimum threshold (positive = eligible)
df['txn_surplus'] = df['transaction_amount_inr'] - df['min_transaction_inr']

# 5. Cashback % × eligibility flag (0 if not eligible)
df['effective_cashback_pct'] = df['cashback_percentage'] * df['offer_applicable']

print('New features added:')
new_feats = ['potential_cashback','capped_potential','txn_above_min','txn_surplus','effective_cashback_pct']
df[new_feats].describe().round(2)


# ## Step 6: Model Training

# In[6]:


# ── 6a. Define target and feature sets ────────────────────────────────────────
TARGET = 'actual_cashback_inr'

# Columns to EXCLUDE from features:
#   - TARGET itself
#   - Leakage columns (computed FROM the target or encoding it directly)
#   - Raw text (offer_description, clean_desc) — already handled by TF-IDF
EXCLUDE = [
    'actual_cashback_inr',    # target
    'effective_savings_pct',  # = actual_cashback / txn_amount  → leakage
    'offer_rank_score',       # derived from effective_savings_pct → leakage
    'offer_description',      # raw text → replaced by TF-IDF
    'clean_desc',             # cleaned text → replaced by TF-IDF
]

struct_cols = [c for c in df.columns if c not in EXCLUDE]
print(f'Structured features ({len(struct_cols)}):', struct_cols)

# ── 6b. Combine structured + TF-IDF features ──────────────────────────────────
X_struct = csr_matrix(df[struct_cols].values.astype(float))
X = hstack([X_struct, tfidf_matrix])          # shape: (3000, len(struct_cols)+50)
y = df[TARGET].values

print(f'\nFinal feature matrix shape: {X.shape}')
print(f'Target range: ₹{y.min():.0f}  →  ₹{y.max():.0f}')


# In[7]:


# ── 6c. Train / Test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f'Train: {X_train.shape[0]} rows   Test: {X_test.shape[0]} rows')


# In[8]:


# ── 6d. Train Random Forest (baseline) ────────────────────────────────────────
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

rf_mae  = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2   = r2_score(y_test, rf_preds)

print('Random Forest Results')
print(f'  MAE  : ₹{rf_mae:.2f}')
print(f'  RMSE : ₹{rf_rmse:.2f}')
print(f'  R²   : {rf_r2:.4f}')


# In[9]:


# ── 6e. Train Gradient Boosting (XGBoost-style, no external lib needed) ───────
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)

gb_mae  = mean_absolute_error(y_test, gb_preds)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_preds))
gb_r2   = r2_score(y_test, gb_preds)

print('Gradient Boosting Results')
print(f'  MAE  : ₹{gb_mae:.2f}')
print(f'  RMSE : ₹{gb_rmse:.2f}')
print(f'  R²   : {gb_r2:.4f}')


# ## Step 7: Evaluation & Accuracy Improvement

# In[10]:


# ── 7a. Model comparison summary ──────────────────────────────────────────────
summary = pd.DataFrame({
    'Model'    : ['Random Forest', 'Gradient Boosting'],
    'MAE (₹)'  : [round(rf_mae, 2),  round(gb_mae, 2)],
    'RMSE (₹)' : [round(rf_rmse, 2), round(gb_rmse, 2)],
    'R² Score' : [round(rf_r2, 4),   round(gb_r2, 4)],
})
print(summary.to_string(index=False))

# Pick best model
best_model  = gb_model if gb_r2 >= rf_r2 else rf_model
best_name   = 'Gradient Boosting' if gb_r2 >= rf_r2 else 'Random Forest'
print(f'\n→ Best model selected: {best_name}')


# In[11]:


# ── 7b. Feature importance (top 15) ──────────────────────────────────────────
# GradientBoosting / RF support .feature_importances_ on dense data.
# We trained on sparse — re-train RF on dense for importance analysis only.

X_dense = pd.DataFrame(
    np.hstack([df[struct_cols].values.astype(float),
               tfidf_matrix.toarray()]),
    columns=struct_cols + list(tfidf.get_feature_names_out())
)

rf_dense = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_dense.fit(X_dense, y)

feat_imp = pd.Series(rf_dense.feature_importances_, index=X_dense.columns)
top15    = feat_imp.nlargest(15)

print('Top 15 Features by Importance:')
print(top15.round(4).to_string())


# In[12]:


# ── 7c. Hyperparameter tuning (RandomizedSearchCV on RF) ─────────────────────
param_grid = {
    'n_estimators' : [100, 200, 300],
    'max_depth'    : [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features' : ['sqrt', 'log2'],
}

rf_tuned = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_grid,
    n_iter=10,
    scoring='r2',
    cv=3,
    random_state=42,
    verbose=0
)
rf_tuned.fit(X_train, y_train)

tuned_preds = rf_tuned.best_estimator_.predict(X_test)
tuned_r2    = r2_score(y_test, tuned_preds)
tuned_mae   = mean_absolute_error(y_test, tuned_preds)
tuned_rmse  = np.sqrt(mean_squared_error(y_test, tuned_preds))

print('Best Params :', rf_tuned.best_params_)
print(f'Tuned RF  →  MAE: ₹{tuned_mae:.2f}  RMSE: ₹{tuned_rmse:.2f}  R²: {tuned_r2:.4f}')

# Update best model if tuned RF is better
if tuned_r2 > gb_r2 and tuned_r2 > rf_r2:
    best_model = rf_tuned.best_estimator_
    best_name  = 'Tuned Random Forest'
print(f'\n→ Final best model: {best_name}')


# In[13]:


# ── 7d. How to further improve model performance ──────────────────────────────
tips = """
How to Improve Model Performance:
──────────────────────────────────
1. Larger dataset        → More training rows reduce overfitting and improve generalization.
2. Real-world data       → Synthetic data has perfect correlations; real data has noise.
3. XGBoost / LightGBM    → Install these libraries for faster, more accurate gradient boosting.
4. More NLP features     → Named-entity extraction (bank name, % value) from offer text.
5. User history features → Avg spend per category, preferred bank, past cashback earned.
6. Cross-validation      → Use 5-fold CV for more robust evaluation.
7. Optuna / GridSearch   → Deeper hyperparameter search with Optuna for auto-optimization.
"""
print(tips)


# ## Step 8: Recommendation System

# In[14]:


# ── 8a. Offer bank & payment catalogue (simulate available offers at inference) 
# In production this comes from a live offer database.
OFFER_CATALOGUE = [
    {'bank':'HDFC',    'payment_method':'Credit Card', 'card_network':'Visa',       'cashback_percentage':10, 'max_cashback_inr':1000, 'min_transaction_inr':10000, 'offer_valid_days':30, 'offer_description':'10% cashback up to Rs.1000 on Electronics using HDFC Credit Card. Min transaction Rs.10000.'},
    {'bank':'ICICI',   'payment_method':'Debit Card',  'card_network':'Mastercard', 'cashback_percentage':7,  'max_cashback_inr':750,  'min_transaction_inr':5000,  'offer_valid_days':30, 'offer_description':'7% cashback up to Rs.750 on Electronics using ICICI Debit Card. Min transaction Rs.5000.'},
    {'bank':'GPay',    'payment_method':'UPI',         'card_network':'N/A',        'cashback_percentage':5,  'max_cashback_inr':200,  'min_transaction_inr':999,   'offer_valid_days':14, 'offer_description':'5% cashback up to Rs.200 on Electronics using GPay UPI. Min transaction Rs.999.'},
    {'bank':'SBI',     'payment_method':'Net Banking', 'card_network':'N/A',        'cashback_percentage':3,  'max_cashback_inr':300,  'min_transaction_inr':2000,  'offer_valid_days':60, 'offer_description':'3% cashback up to Rs.300 on Electronics using SBI Net Banking. Min transaction Rs.2000.'},
    {'bank':'Paytm',   'payment_method':'Wallet',      'card_network':'N/A',        'cashback_percentage':8,  'max_cashback_inr':150,  'min_transaction_inr':500,   'offer_valid_days':7,  'offer_description':'8% cashback up to Rs.150 on Electronics using Paytm Wallet. Min transaction Rs.500.'},
    {'bank':'Axis',    'payment_method':'Credit Card', 'card_network':'Visa',       'cashback_percentage':12, 'max_cashback_inr':500,  'min_transaction_inr':8000,  'offer_valid_days':30, 'offer_description':'12% cashback up to Rs.500 on Electronics using Axis Credit Card. Min transaction Rs.8000.'},
    {'bank':'Kotak',   'payment_method':'Debit Card',  'card_network':'RuPay',      'cashback_percentage':5,  'max_cashback_inr':400,  'min_transaction_inr':3000,  'offer_valid_days':30, 'offer_description':'5% cashback up to Rs.400 on Electronics using Kotak Debit Card. Min transaction Rs.3000.'},
    {'bank':'PhonePe', 'payment_method':'UPI',         'card_network':'N/A',        'cashback_percentage':6,  'max_cashback_inr':250,  'min_transaction_inr':1000,  'offer_valid_days':14, 'offer_description':'6% cashback up to Rs.250 on Electronics using PhonePe UPI. Min transaction Rs.1000.'},
]
print(f'Offer catalogue loaded: {len(OFFER_CATALOGUE)} offers available')


# In[15]:


# ── 8b. Recommendation function ───────────────────────────────────────────────
def recommend_best_offer(
    transaction_amount,
    category,
    platform,
    user_spending_profile,
    user_bank_preference,
    top_n=3
):
    """
    Given a transaction, predicts cashback for every offer in the catalogue,
    ranks them, and returns the top_n recommendations with explanations.
    """
    results = []

    for offer in OFFER_CATALOGUE:
        # Build a single-row DataFrame matching training columns
        row = {
            'user_spending_profile' : user_spending_profile,
            'user_bank_preference'  : user_bank_preference,
            'platform'              : platform,
            'category'              : category,
            'transaction_amount_inr': transaction_amount,
            'payment_method'        : offer['payment_method'],
            'bank'                  : offer['bank'],
            'card_network'          : offer['card_network'],
            'cashback_percentage'   : offer['cashback_percentage'],
            'max_cashback_inr'      : offer['max_cashback_inr'],
            'min_transaction_inr'   : offer['min_transaction_inr'],
            'offer_valid_days'      : offer['offer_valid_days'],
        }

        # Eligibility check
        row['offer_applicable'] = int(transaction_amount >= offer['min_transaction_inr'])

        # Derived features (same as training)
        row['potential_cashback']    = (offer['cashback_percentage'] / 100) * transaction_amount
        row['capped_potential']      = min(row['potential_cashback'], offer['max_cashback_inr'])
        row['txn_above_min']         = row['offer_applicable']
        row['txn_surplus']           = transaction_amount - offer['min_transaction_inr']
        row['effective_cashback_pct']= offer['cashback_percentage'] * row['offer_applicable']

        # Encode categoricals (use same LabelEncoders from training)
        for col in cat_cols:
            le = le_dict[col]
            val = str(row[col])
            if val in le.classes_:
                row[col] = le.transform([val])[0]
            else:
                row[col] = 0   # unseen label → default 0

        # Build structured feature row
        struct_row = np.array([[row[c] for c in struct_cols]], dtype=float)

        # TF-IDF on offer description
        nlp_row = tfidf.transform([offer['offer_description']])

        # Combine
        X_pred = hstack([csr_matrix(struct_row), nlp_row])

        # Predict cashback
        predicted_cb = max(0, best_model.predict(X_pred)[0])   # clamp to ≥ 0

        # If not eligible, cashback = 0
        if not row['offer_applicable']:
            predicted_cb = 0

        results.append({
            'bank'                  : offer['bank'],
            'payment_method'        : offer['payment_method'],
            'card_network'          : offer['card_network'],
            'cashback_percentage'   : offer['cashback_percentage'],
            'max_cashback_inr'      : offer['max_cashback_inr'],
            'min_transaction_inr'   : offer['min_transaction_inr'],
            'offer_applicable'      : bool(row['offer_applicable']),
            'predicted_cashback_inr': round(predicted_cb, 2),
            'effective_savings_pct' : round((predicted_cb / transaction_amount) * 100, 2),
        })

    # Sort by predicted cashback descending
    results_df = pd.DataFrame(results).sort_values(
        'predicted_cashback_inr', ascending=False
    ).reset_index(drop=True)

    results_df.insert(0, 'rank', results_df.index + 1)
    results_df['is_best_offer'] = results_df['rank'] == 1

    return results_df.head(top_n)

print('Recommendation function defined.')


# In[16]:


# ── 8c. Run a sample recommendation ──────────────────────────────────────────
TRANSACTION = {
    'transaction_amount'   : 12500,
    'category'             : 'Electronics',
    'platform'             : 'Amazon',
    'user_spending_profile': 'High Spender',
    'user_bank_preference' : 'HDFC',
}

top3 = recommend_best_offer(
    transaction_amount    = TRANSACTION['transaction_amount'],
    category              = TRANSACTION['category'],
    platform              = TRANSACTION['platform'],
    user_spending_profile = TRANSACTION['user_spending_profile'],
    user_bank_preference  = TRANSACTION['user_bank_preference'],
    top_n=3
)

print('Top 3 Recommended Offers:')
print(top3[[
    'rank','bank','payment_method','cashback_percentage',
    'max_cashback_inr','predicted_cashback_inr','effective_savings_pct','offer_applicable','is_best_offer'
]].to_string(index=False))


# ## Step 9: Explanation Engine (Human-Readable Output)

# In[17]:


def explain_recommendation(transaction_amount, category, platform,
                           user_spending_profile, user_bank_preference, top_n=3):
    """
    Prints a friendly, non-technical explanation of the top offers.
    """
    results = recommend_best_offer(
        transaction_amount, category, platform,
        user_spending_profile, user_bank_preference, top_n=top_n
    )

    best = results.iloc[0]

    print('=' * 58)
    print('        OptiPay AI — Payment Recommendation')
    print('=' * 58)
    print(f'  Transaction  : ₹{transaction_amount:,.0f}')
    print(f'  Category     : {category}')
    print(f'  Platform     : {platform}')
    print('=' * 58)

    print('\n  BEST PAYMENT METHOD:')
    print(f'  ★  {best["bank"]} {best["payment_method"]}')
    if best['card_network'] != 'N/A':
        print(f'     Card Network : {best["card_network"]}')
    print(f'     Expected Cashback : ₹{best["predicted_cashback_inr"]:,.2f}')
    print(f'     Effective Savings : {best["effective_savings_pct"]}% of transaction')

    print('\n  WHY THIS OFFER?')
    raw_cb = (best['cashback_percentage'] / 100) * transaction_amount
    print(f'  • Cashback rate      : {best["cashback_percentage"]}%')
    print(f'  • Raw cashback       : ₹{raw_cb:,.0f}  (before cap)')
    print(f'  • Maximum cap        : ₹{best["max_cashback_inr"]:,.0f}')
    print(f'  • Min txn required   : ₹{best["min_transaction_inr"]:,.0f}')
    eligible_str = 'Yes ✓' if best['offer_applicable'] else 'No ✗'
    print(f'  • Eligible           : {eligible_str}')
    if raw_cb > best['max_cashback_inr']:
        print(f'  • Note               : Cashback capped at ₹{best["max_cashback_inr"]:,.0f}')

    print('\n  TOP 3 OFFERS COMPARISON:')
    print(f'  {"#":<3} {"Bank & Method":<28} {"Cashback":>10}  {"Savings %":>9}')
    print('  ' + '-' * 54)
    for _, row in results.iterrows():
        star     = '★' if row['is_best_offer'] else ' '
        label    = f"{row['bank']} {row['payment_method']}"
        cb_str   = f"₹{row['predicted_cashback_inr']:,.0f}" if row['offer_applicable'] else 'Not eligible'
        pct_str  = f"{row['effective_savings_pct']}%" if row['offer_applicable'] else '—'
        print(f'  {star}{row["rank"]:<3} {label:<28} {cb_str:>10}  {pct_str:>9}')

    print('\n' + '=' * 58)


# ── Run the explanation ───────────────────────────────────────────────────────
explain_recommendation(
    transaction_amount    = 12500,
    category              = 'Electronics',
    platform              = 'Amazon',
    user_spending_profile = 'High Spender',
    user_bank_preference  = 'HDFC',
    top_n=3
)


# In[ ]:


# ── Test with another transaction ─────────────────────────────────────────────
explain_recommendation(
    transaction_amount    = 650,
    category              = 'Food Delivery',
    platform              = 'Swiggy',
    user_spending_profile = 'Frequent Buyer',
    user_bank_preference  = 'No Preference',
    top_n=3
)


# ---
# ## Summary
# 
# | Step | What was done |
# |------|---------------|
# | 1    | Loaded `optipay_ai_dataset.csv` |
# | 2    | Dropped IDs/dates, filled `card_network` nulls, converted boolean |
# | 3    | Label-encoded 7 categorical columns (stored encoders for reuse) |
# | 4    | Cleaned offer descriptions → TF-IDF (50 features, bigrams) |
# | 5    | Added 5 derived features: `potential_cashback`, `capped_potential`, `txn_above_min`, `txn_surplus`, `effective_cashback_pct` |
# | 6    | Trained Random Forest + Gradient Boosting on combined structured+NLP features |
# | 7    | Evaluated with MAE / RMSE / R² — selected best model automatically |
# | 8    | Recommendation engine scores all catalogue offers for any transaction |
# | 9    | Human-readable explanation with reason, comparison table, eligibility check |
# 
