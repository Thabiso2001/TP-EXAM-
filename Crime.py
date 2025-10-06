
import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Try statsmodels for forecasting
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS = True
except Exception:
    STATSMODELS = False

st.set_page_config(page_title="Crime Analysis & Forecast (Exam)", layout="wide")

# ---------- Helpers ----------
def read_file(uploaded):
    if uploaded is None:
        return None
    name = getattr(uploaded, "name", "")
    try:
        if name.endswith(".csv") or name.endswith(".txt"):
            return pd.read_csv(uploaded, low_memory=False)
        elif name.endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded)
        elif name.endswith(".parquet"):
            return pd.read_parquet(uploaded)
        else:
            # try csv fallback
            uploaded.seek(0)
            return pd.read_csv(uploaded, low_memory=False)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

def detect_and_parse_date(df):
    """
    Tries multiple strategies to create/parse a 'date' column:
    1. If any column contains 'date' -> try parse
    2. If columns like Year & Month exist -> combine into first-of-month
    3. If column 'period' looks like 'YYYY-MM' or 'YYYY/Q1' -> parse
    Returns (df, date_col_name or None)
    """
    df = df.copy()
    cols = [c for c in df.columns]
    # 1: find columns with 'date' in name
    date_candidates = [c for c in cols if "date" in c.lower()]
    if date_candidates:
        c = date_candidates[0]
        parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        if parsed.notna().any():
            df["date"] = parsed
            return df, "date"
    # 2: Year+Month or Year+Quarter
    year_cols = [c for c in cols if c.lower() in ("year", "yr")]
    month_cols = [c for c in cols if c.lower() in ("month", "mnth", "mth")]
    if year_cols and month_cols:
        y = df[year_cols[0]].astype(str).str.extract(r'(\d{4})', expand=False)
        m = df[month_cols[0]].astype(str).str.extract(r'(\d{1,2})', expand=False)
        # fill missing month with 1
        m = m.fillna("1")
        combined = y + "-" + m + "-01"
        parsed = pd.to_datetime(combined, errors="coerce", dayfirst=True)
        if parsed.notna().any():
            df["date"] = parsed
            return df, "date"
    # 3: 'period' or 'yyyymm' like fields
    period_cols = [c for c in cols if "period" in c.lower() or "yyy" in c.lower() or "month" in c.lower() and "year" in c.lower()]
    if period_cols:
        c = period_cols[0]
        s = df[c].astype(str)
        # try parse YYYY-MM or YYYY/MM or YYYYMM
        parsed = pd.to_datetime(s.str.replace("/", "-").str.replace(r"(\d{4})(\d{2})", r"\1-\2"), errors="coerce", dayfirst=True)
        if parsed.notna().any():
            df["date"] = parsed
            return df, "date"
    # 4: try any column that looks datetime after coercion
    for c in cols:
        parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        if parsed.notna().any():
            df["date"] = parsed
            return df, "date"
    return df, None

def basic_relevance_check(df):
    checks = {}
    checks["n_rows"] = len(df)
    checks["n_columns"] = len(df.columns)
    expected_keywords = ["date","category","location","lat","lon","latitude","longitude","crime","offence"]
    present = []
    for k in expected_keywords:
        for c in df.columns:
            if k in c.lower():
                present.append(c)
                break
    checks["expected_like_cols_found_sample"] = present
    # date range if possible
    if "date" in df.columns:
        try:
            checks["date_range"] = (str(df["date"].min()), str(df["date"].max()))
        except Exception:
            checks["date_range"] = None
    else:
        checks["date_range"] = None
    return checks

def plot_eda(df, date_col, category_col, location_col, lat_col=None, lon_col=None):
    st.subheader("Exploratory Data Analysis")
    c1, c2 = st.columns((2,1))
    with c1:
        st.markdown("**Crimes over time (monthly counts)**")
        by_month = df.set_index(date_col).resample("M").size().rename("count").reset_index()
        fig = px.line(by_month, x=date_col, y="count", title="Monthly crime counts")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Top categories**")
        topcats = df[category_col].value_counts().reset_index().rename(columns={"index":category_col, category_col:"count"})
        st.plotly_chart(px.bar(topcats.head(20), x=category_col, y="count", title="Top categories"), use_container_width=True)

    with c2:
        st.markdown("**Location vs Category heatmap (top 10 each)**")
        topc = df[category_col].value_counts().head(10).index
        topl = df[location_col].value_counts().head(10).index
        pivot = pd.pivot_table(df[df[category_col].isin(topc) & df[location_col].isin(topl)],
                               index=location_col, columns=category_col,
                               values=date_col, aggfunc="count", fill_value=0)
        if pivot.shape[0] and pivot.shape[1]:
            fig2 = px.imshow(pivot, labels=dict(x="Category", y="Location", color="Count"), aspect="auto")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Not enough data to create heatmap for top categories/locations.")

    if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
        st.markdown("**Map (sampled incidents)**")
        sample = df[[lat_col, lon_col, category_col, date_col]].dropna().sample(min(2000, len(df)))
        fig3 = px.scatter_mapbox(sample, lat=lat_col, lon=lon_col, hover_data=[category_col, date_col], zoom=9)
        fig3.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig3, use_container_width=True)

def run_classification(df, target_col, exclude_cols, test_size=0.2, random_state=42):
    st.subheader("Classification (Random Forest)")
    df = df.dropna(subset=[target_col])
    # basic features: numeric + small-cardinality categoricals chosen automatically
    features = [c for c in df.columns if c not in exclude_cols + [target_col, "date"]]
    if not features:
        st.warning("No features available for classification. Select columns or provide features.")
        return None, None
    X = df[features].copy()
    y = df[target_col].astype(str)
    # simple preprocessing
    for c in X.columns:
        if X[c].dtype.kind in "biufc":
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].fillna("missing").astype(str)
    X = pd.get_dummies(X, drop_first=True)
    if X.shape[0] < 10 or len(y.unique()) < 2:
        st.warning("Not enough data or target classes for classification.")
        return None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: **{acc:.3f}**")
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
    # classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())
    # feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(30)
    st.subheader("Top feature importances")
    st.bar_chart(importances)
    return model, X.columns

def forecast_series(series, periods=30):
    """
    Input: pandas Series indexed by datetime representing daily counts
    Returns: pred (Series), lower (Series), upper (Series)
    """
    series = series.asfreq("D").fillna(0)
    if STATSMODELS and len(series) > 14:
        try:
            model = ExponentialSmoothing(series, seasonal="add", seasonal_periods=7).fit(optimized=True)
            pred = model.forecast(periods)
            resid = model.resid.dropna()
            sigma = resid.std() if len(resid) > 1 else series.std()
            ci = 1.96 * sigma
            upper = pred + ci
            lower = pred - ci
            return pred, lower, upper
        except Exception:
            pass
    # fallback: naive mean + residual-based CI
    mean = series.mean()
    pred = pd.Series([mean]*periods, index=pd.date_range(start=series.index[-1] + pd.Timedelta(1, "D"), periods=periods))
    sigma = series.diff().std()
    if np.isnan(sigma):
        sigma = series.std() if series.std() > 0 else 1.0
    ci = 1.96 * sigma
    upper = pred + ci
    lower = pred - ci
    return pred, lower, upper

# ---------- App Layout ----------
st.title("Crime Analysis & Forecast — FINAL_EXAM2025")

st.markdown("Upload a crime dataset (CSV/Excel/Parquet). Expected useful columns: date (or year/month), category, location, latitude, longitude, plus any explanatory features.")

uploaded = st.file_uploader("Upload dataset", type=["csv","xlsx","xls","parquet","txt"])
use_sample = st.checkbox("Use built-in synthetic sample dataset (for testing)", value=False)

df = None
if use_sample and uploaded is None:
    # generate synthetic data quickly
    n = 3000
    rng = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="H")
    df = pd.DataFrame({
        "date": np.random.choice(rng, n),
        "category": np.random.choice(["Theft","Assault","Burglary","Robbery","Vandalism"], n),
        "location": np.random.choice(["Central","East","West","North","South"], n),
        "latitude": np.random.uniform(-33.95, -33.80, n),
        "longitude": np.random.uniform(18.40, 18.70, n),
        "feature1": np.random.randn(n),
        "feature2": np.random.randint(0,5,n)
    })
elif uploaded is not None:
    df = read_file(uploaded)

if df is None:
    st.info("Please upload a dataset or enable the sample dataset box to test the app.")
    st.stop()

# Try to detect date and parse
df, detected_date_col = detect_and_parse_date(df)
if detected_date_col is None:
    st.error("Date column not found or could not be parsed. Please provide a date-like column (e.g., 'date', 'reported_date', or Year & Month columns).")
    # show a helpful preview of candidate columns
    st.write("Columns found in your file:", list(df.columns))
    st.stop()

# now we have a date column
date_col = "date"
# Force timezone-naive datetimes
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
if df[date_col].isna().all():
    st.error("Failed to convert any values to datetime in the detected date column. Please check formats (YYYY-MM-DD, DD/MM/YYYY, or Year+Month).")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters & Settings")
min_date = df[date_col].min().date()
max_date = df[date_col].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# detect category and location columns heuristically
cols = df.columns.tolist()
cat_candidates = [c for c in cols if "category" in c.lower() or "crime" in c.lower() or "offence" in c.lower() or "type" in c.lower()]
loc_candidates = [c for c in cols if "location" in c.lower() or "place" in c.lower() or "area" in c.lower() or "suburb" in c.lower() or "district" in c.lower()]

category_col = st.sidebar.selectbox("Category column", options=cat_candidates + ["--none--"], index=0) if cat_candidates else st.sidebar.selectbox("Category column", options=cols, index=0)
location_col = st.sidebar.selectbox("Location column", options=loc_candidates + ["--none--"], index=0) if loc_candidates else st.sidebar.selectbox("Location column", options=cols, index=1 if len(cols)>1 else 0)

# lat/lon optional
lat_col = st.sidebar.selectbox("Latitude column (optional)", options=["--none--"] + cols, index=0)
lon_col = st.sidebar.selectbox("Longitude column (optional)", options=["--none--"] + cols, index=0)

# apply filters
start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
mask = (df[date_col] >= start_dt) & (df[date_col] <= end_dt)

# when user marked "--none--" convert to generic
if category_col == "--none--":
    if "category" in df.columns:
        category_col = "category"
    else:
        # pick a sensible string column for category
        string_cols = [c for c in df.columns if df[c].dtype == object]
        category_col = string_cols[0] if string_cols else df.columns[0]
if location_col == "--none--":
    if "location" in df.columns:
        location_col = "location"
    else:
        string_cols = [c for c in df.columns if df[c].dtype == object and c != category_col]
        location_col = string_cols[0] if string_cols else df.columns[0]

# restrict to user selections for filters (show all by default)
all_categories = sorted(df[category_col].dropna().unique()) if category_col in df.columns else []
selected_categories = st.sidebar.multiselect("Categories", options=all_categories, default=all_categories if len(all_categories)<=10 else all_categories[:10])
all_locations = sorted(df[location_col].dropna().unique()) if location_col in df.columns else []
selected_locations = st.sidebar.multiselect("Locations", options=all_locations, default=all_locations if len(all_locations)<=10 else all_locations[:10])

if selected_categories:
    mask &= df[category_col].isin(selected_categories)
if selected_locations:
    mask &= df[location_col].isin(selected_locations)

df_filt = df.loc[mask].copy()
st.write(f"Filtered dataset: **{len(df_filt):,}** rows — Date range: **{start_dt.date()}** to **{end_dt.date()}**")

# Relevance checks
st.sidebar.header("File relevance checks")
checks = basic_relevance_check(df)
for k,v in checks.items():
    st.sidebar.write(f"**{k}**: {v}")

# EDA
plot_eda(df_filt, date_col, category_col, location_col, lat_col if lat_col != "--none--" else None, lon_col if lon_col != "--none--" else None)

# Classification
st.header("Classification & Evaluation")
st.markdown("Choose feature columns for classification. Non-numeric columns will be one-hot encoded.")
feature_candidates = [c for c in df_filt.columns if c not in [date_col, category_col, location_col]]
selected_features = st.multiselect("Feature columns", options=feature_candidates, default=feature_candidates[:6])

if st.button("Run classification"):
    if category_col not in df_filt.columns:
        st.error("Category column not found in filtered data.")
    elif not selected_features:
        st.error("Select at least one feature column.")
    else:
        try:
            model, feat_cols = run_classification(df_filt, category_col, exclude_cols=[location_col] if location_col else [], test_size=0.2)
        except Exception as e:
            st.error(f"Classification failed: {e}")

# Forecasting
st.header("Time-series Forecasting")
st.markdown("Select grouping to create daily series (counts) and forecast future counts.")
group_by = st.selectbox("Group incidents by", options=["None", "Category", "Location"], index=0)
if group_by == "Category":
    group_choice = st.selectbox("Choose category for forecast", options=sorted(df_filt[category_col].unique()))
    series = df_filt[df_filt[category_col] == group_choice].set_index(date_col).resample("D").size()
elif group_by == "Location":
    group_choice = st.selectbox("Choose location for forecast", options=sorted(df_filt[location_col].unique()))
    series = df_filt[df_filt[location_col] == group_choice].set_index(date_col).resample("D").size()
else:
    series = df_filt.set_index(date_col).resample("D").size()

st.line_chart(series.rename("count"))

horizon = st.number_input("Forecast horizon (days)", min_value=7, max_value=365, value=30)
if st.button("Run forecast"):
    if len(series) < 10:
        st.warning("Not enough history to generate a reliable forecast.")
    else:
        pred, lower, upper = forecast_series(series, periods=horizon)
        # plotting
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name="historical"))
        fig.add_trace(go.Scatter(x=pred.index, y=pred.values, name="forecast"))
        fig.add_trace(go.Scatter(x=pred.index, y=upper.values, name="upper", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=pred.index, y=lower.values, name="lower", line=dict(dash="dash")))
        fig.update_layout(title="Forecast with approx. 95% CI", xaxis_title="Date", yaxis_title="Daily counts")
        st.plotly_chart(fig, use_container_width=True)

# Summaries
st.header("Summaries")
with st.expander("Technical summary"):
    st.markdown("""
    - Dataset size, date range, and columns shown above.
    - Filters applied: categories, locations, date range.
    - EDA: monthly trend, top categories, heatmap (locations x categories), map (sample).
    - Classification: RandomForest used with basic preprocessing (median imputation + one-hot encoding). Metrics shown: accuracy, confusion matrix, precision/recall/F1.
    - Forecasting: ExponentialSmoothing (Holt-Winters) used when available; fallback to simple mean forecast. Prediction intervals are computed from residual standard deviation (approx. 95% CI = ±1.96*σ_resid).
    - Reproducibility: Save filtered dataset and document exact column choices.
    """)

with st.expander("Non-technical summary"):
    st.markdown("""
    - This dashboard shows how crimes change over time and where they are concentrated.
    - Use the filters on the left to focus on specific crime types, places, and time periods.
    - A simple machine learning model attempts to predict crime type from selected features — results are displayed as accuracy and a confusion matrix.
    - Forecasts provide a short-term projection of daily incident counts with an uncertainty band to indicate possible range.
    - Use these tools for exploratory analysis and to inform further investigation or resource planning.
    """)

# Export
st.header("Export & Notes")
if st.button("Download filtered dataset as CSV"):
    csv = df_filt.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="filtered_crime.csv", mime="text/csv")

st.markdown("**Notes & assumptions:**")
st.markdown("""
- This app makes simplifying assumptions (basic imputation, automatic one-hot encoding).
- For production use: clean & standardize categories/locations, engineer features, perform model selection & cross-validation, and validate forecasts with holdout sets.
- If your dataset uses separate Year and Month columns or a non-standard date string, the app attempts to create/parse a 'date' column automatically.
""")
