import streamlit as st
import pandas as pd
import numpy as np
import gc
import time
import psutil
from datetime import timedelta
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.isotonic import IsotonicRegression
from betacal import BetaCalibration
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import shap

st.set_page_config("🏆 MLB Home Run Predictor — State of the Art, Full Phase 1", layout="wide")
st.title("🏆 MLB Home Run Predictor — State of the Art, Full Phase 1")

@st.cache_data(show_spinner=False, max_entries=2)
def safe_read_cached(path):
    fn = str(getattr(path, 'name', path)).lower()
    if fn.endswith('.parquet'):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin1', low_memory=False)

def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def find_duplicate_columns(df):
    return [col for col in df.columns if list(df.columns).count(col) > 1]

def fix_types(df):
    for col in df.columns:
        if df[col].isnull().all():
            continue
        if df[col].dtype == 'O':
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception:
                pass
        if pd.api.types.is_float_dtype(df[col]) and (df[col].dropna() % 1 == 0).all():
            df[col] = df[col].astype(pd.Int64Dtype())
    return df

def clean_X(df, train_cols=None):
    df = dedup_columns(df)
    df = fix_types(df)
    allowed_obj = {'wind_dir_string', 'condition', 'player_name', 'city', 'park', 'roof_status', 'team_code', 'time'}
    drop_cols = [c for c in df.select_dtypes('O').columns if c not in allowed_obj]
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df.fillna(-1)
    if train_cols is not None:
        for c in train_cols:
            if c not in df.columns:
                df[c] = -1
        df = df[list(train_cols)]
    return df

def get_valid_feature_cols(df, drop=None):
    base_drop = set(['game_date','batter_id','player_name','pitcher_id','city','park','roof_status','team_code','time'])
    if drop: base_drop = base_drop.union(drop)
    numerics = df.select_dtypes(include=[np.number]).columns
    return [c for c in numerics if c not in base_drop]

def nan_inf_check(X, name):
    if isinstance(X, pd.DataFrame):
        X_num = X.select_dtypes(include=[np.number])
        nans = X_num.isna().sum().sum()
        infs = np.isinf(X_num.to_numpy(dtype=np.float64, copy=False)).sum()
    else:
        nans = np.isnan(X).sum()
        infs = np.isinf(X).sum()
    if nans > 0 or infs > 0:
        st.error(f"Found {nans} NaNs and {infs} Infs in {name}! Please fix.")
        st.stop()

def feature_debug(X):
    X.columns = X.columns.astype(str)  # Ensure column names are strings

    st.write("🛡️ Feature Debugging:")
    st.write("Data types:", X.dtypes.value_counts())

    object_cols = X.select_dtypes(include="object").columns.tolist()
    if object_cols:
        st.write("Columns with object dtype:", object_cols)
    else:
        st.write("Columns with object dtype: []")

    for col in X.columns:
        try:
            if X[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                st.write(f"Column `{col}` is {X[col].dtype}, unique values: {X[col].unique()[:8]}")
        except Exception as e:
            st.write(f"⚠️ Could not inspect column `{col}`: {e}")

    st.write("Missing values per column (top 10):", X.isna().sum().sort_values(ascending=False).head(10))
def overlay_multiplier(row):
    """
    Upgraded overlay multiplier using:
    - Batted ball pull/oppo/fb/gb/air for both batter and pitcher.
    - Full wind direction parsing.
    - Handedness logic.
    - Amplified, but reasonable, effects for correct context.
    """
    edge = 1.0

    # --- Get relevant values safely ---
    wind = row.get("wind_mph", np.nan)
    wind_dir = str(row.get("wind_dir_string", "")).lower().strip()
    temp = row.get("temp", np.nan)
    humidity = row.get("humidity", np.nan)
    park_hr_col = 'park_hr_rate'

    # Batter
    b_hand = str(row.get('stand', row.get('batter_hand', 'R'))).upper() or "R"
    b_pull = row.get("pull_rate", np.nan)
    b_oppo = row.get("oppo_rate", np.nan)
    b_fb = row.get("fb_rate", np.nan)
    b_air = row.get("air_rate", np.nan)
    b_ld = row.get("ld_rate", np.nan)
    b_pu = row.get("pu_rate", np.nan)
    b_hot = row.get("b_hr_per_pa_7", np.nan)  # rolling HR/PA
    
    # Pitcher
    p_hand = str(row.get("pitcher_hand", "")).upper() or "R"
    p_fb = row.get("p_fb_rate", row.get("fb_rate", np.nan))
    p_gb = row.get("p_gb_rate", row.get("gb_rate", np.nan))
    p_air = row.get("p_air_rate", row.get("air_rate", np.nan))
    p_ld = row.get("p_ld_rate", row.get("ld_rate", np.nan))
    p_pu = row.get("p_pu_rate", row.get("pu_rate", np.nan))

    # --- Wind logic: Amplified/Smart ---
    wind_factor = 1.0
    if wind is not None and pd.notnull(wind) and wind >= 7 and wind_dir and wind_dir != "nan":
        # Strong outfield wind: boost for right context
        for field, out_bonus, in_bonus, field_side in [
            ("rf", 1.19, 0.85, ("R", "oppo")),  # RHH oppo or LHH pull to RF
            ("lf", 1.19, 0.85, ("R", "pull")),  # RHH pull or LHH oppo to LF
            ("cf", 1.11, 0.90, ("ANY", "fb")),  # Any FB to CF
        ]:
            if field in wind_dir:
                if "out" in wind_dir or "o" in wind_dir:
                    if field == "rf":
                        if (b_hand == "R" and b_oppo > 0.26) or (b_hand == "L" and b_pull > 0.35):
                            wind_factor *= out_bonus
                    elif field == "lf":
                        if (b_hand == "R" and b_pull > 0.35) or (b_hand == "L" and b_oppo > 0.26):
                            wind_factor *= out_bonus
                    elif field == "cf":
                        if b_fb > 0.21 or b_air > 0.34:
                            wind_factor *= out_bonus
                if "in" in wind_dir or "i" in wind_dir:
                    if field == "rf":
                        if (b_hand == "R" and b_oppo > 0.26) or (b_hand == "L" and b_pull > 0.35):
                            wind_factor *= in_bonus
                    elif field == "lf":
                        if (b_hand == "R" and b_pull > 0.35) or (b_hand == "L" and b_oppo > 0.26):
                            wind_factor *= in_bonus
                    elif field == "cf":
                        if b_fb > 0.21 or b_air > 0.34:
                            wind_factor *= in_bonus

        # Add extra boost/fade for high-flyball hitters facing high-flyball pitchers
        if p_fb is not np.nan and p_fb > 0.25 and (b_fb > 0.23 or b_air > 0.36):
            if "out" in wind_dir or "o" in wind_dir:
                wind_factor *= 1.09
            elif "in" in wind_dir or "i" in wind_dir:
                wind_factor *= 0.94
        # Fade for extreme groundball pitchers
        if p_gb is not np.nan and p_gb > 0.53:
            wind_factor *= 0.93

    # --- Hot streak logic (recent HR/PA) ---
    if b_hot is not np.nan and b_hot > 0.09:
        edge *= 1.04
    elif b_hot is not np.nan and b_hot < 0.025:
        edge *= 0.97

    # --- Weather logic ---
    if temp is not None and pd.notnull(temp):
        edge *= 1.036 ** ((temp - 70) / 10)
    if humidity is not None and pd.notnull(humidity):
        if humidity > 65: edge *= 1.02
        elif humidity < 35: edge *= 0.98

    # --- Park HR Rate logic ---
    if park_hr_col in row and pd.notnull(row[park_hr_col]):
        pf = max(0.80, min(1.22, float(row[park_hr_col])))
        edge *= pf

    # --- Multiply wind after everything else, so it can be "amplifying" ---
    edge *= wind_factor

    # --- Clamp for sanity ---
    return float(np.clip(edge, 0.70, 1.36))
    def apply_wind_bonus(factor, cond):
        nonlocal edge
        if cond:
            edge *= factor

    if wind >= 7 and isinstance(wind_dir, str) and wind_dir and wind_dir != "nan":
        if "out" in wind_dir or "o" in wind_dir:
            if "rf" in wind_dir:
                apply_wind_bonus(1.10, hand == "R" and b_oppo is not np.nan and b_oppo > 0.28)
                apply_wind_bonus(1.13, hand == "L" and b_pull is not np.nan and b_pull > 0.37)
            if "lf" in wind_dir:
                apply_wind_bonus(1.13, hand == "R" and b_pull is not np.nan and b_pull > 0.37)
                apply_wind_bonus(1.10, hand == "L" and b_oppo is not np.nan and b_oppo > 0.28)
            if "cf" in wind_dir:
                apply_wind_bonus(1.06, b_fb is not np.nan and b_fb > 0.22)
        elif "in" in wind_dir or "i" in wind_dir:
            if "rf" in wind_dir:
                apply_wind_bonus(0.91, hand == "R" and b_oppo is not np.nan and b_oppo > 0.28)
                apply_wind_bonus(0.88, hand == "L" and b_pull is not np.nan and b_pull > 0.37)
            if "lf" in wind_dir:
                apply_wind_bonus(0.88, hand == "R" and b_pull is not np.nan and b_pull > 0.37)
                apply_wind_bonus(0.91, hand == "L" and b_oppo is not np.nan and b_oppo > 0.28)
            if "cf" in wind_dir:
                apply_wind_bonus(0.93, b_fb is not np.nan and b_fb > 0.22)

        if p_fb is not np.nan and p_fb > 0.24:
            if "out" in wind_dir or "o" in wind_dir:
                edge *= 1.05
            elif "in" in wind_dir or "i" in wind_dir:
                edge *= 0.97
        if p_gb is not np.nan and p_gb > 0.49:
            edge *= 0.97

    temp = row.get("temp", np.nan)
    if pd.notnull(temp):
        edge *= 1.03 ** ((temp - 70) / 10)

    humidity = row.get("humidity", np.nan)
    if pd.notnull(humidity):
        if humidity > 60:
            edge *= 1.02
        elif humidity < 40:
            edge *= 0.98

    park_hr_col = 'park_hr_rate'
    if park_hr_col in row and pd.notnull(row[park_hr_col]):
        pf = max(0.85, min(1.20, float(row[park_hr_col])))
        edge *= pf

    return float(np.clip(edge, 0.75, 1.33))

def rate_weather(row):
    ratings = {}
    temp = row.get("temp", np.nan)
    if pd.isna(temp):
        ratings["temp_rating"] = "?"
    elif 68 <= temp <= 85:
        ratings["temp_rating"] = "Excellent"
    elif 60 <= temp < 68 or 85 < temp <= 92:
        ratings["temp_rating"] = "Good"
    elif 50 <= temp < 60 or 92 < temp <= 98:
        ratings["temp_rating"] = "Fair"
    else:
        ratings["temp_rating"] = "Poor"
    humidity = row.get("humidity", np.nan)
    if pd.isna(humidity):
        ratings["humidity_rating"] = "?"
    elif 45 <= humidity <= 65:
        ratings["humidity_rating"] = "Excellent"
    elif 30 <= humidity < 45 or 65 < humidity <= 80:
        ratings["humidity_rating"] = "Good"
    elif 15 <= humidity < 30 or 80 < humidity <= 90:
        ratings["humidity_rating"] = "Fair"
    else:
        ratings["humidity_rating"] = "Poor"
    wind = row.get("wind_mph", np.nan)
    wind_dir = str(row.get("wind_dir_string", "")).lower()
    if pd.isna(wind):
        ratings["wind_rating"] = "?"
    elif wind < 6:
        ratings["wind_rating"] = "Excellent"
    elif 6 <= wind < 12:
        ratings["wind_rating"] = "Good"
    elif 12 <= wind < 18:
        if "out" in wind_dir:
            ratings["wind_rating"] = "Good"
        elif "in" in wind_dir:
            ratings["wind_rating"] = "Fair"
        else:
            ratings["wind_rating"] = "Fair"
    else:
        if "out" in wind_dir:
            ratings["wind_rating"] = "Fair"
        elif "in" in wind_dir:
            ratings["wind_rating"] = "Poor"
        else:
            ratings["wind_rating"] = "Poor"
    condition = str(row.get("condition", "")).lower()
    if "clear" in condition or "sun" in condition or "outdoor" in condition:
        ratings["condition_rating"] = "Excellent"
    elif "cloud" in condition or "partly" in condition:
        ratings["condition_rating"] = "Good"
    elif "rain" in condition or "fog" in condition:
        ratings["condition_rating"] = "Poor"
    else:
        ratings["condition_rating"] = "Fair"
    return pd.Series(ratings)

def drift_check(train, today, n=5):
    drifted = []
    for c in train.columns:
        if c not in today.columns: continue
        tmean = np.nanmean(train[c])
        tstd = np.nanstd(train[c])
        dmean = np.nanmean(today[c])
        if tstd > 0 and abs(tmean - dmean) / tstd > n:
            drifted.append(c)
    return drifted

def winsorize_clip(X, limits=(0.01, 0.99)):
    X = X.astype(float)
    for col in X.columns:
        lower = X[col].quantile(limits[0])
        upper = X[col].quantile(limits[1])
        X[col] = X[col].clip(lower=lower, upper=upper)
    return X

def stickiness_rank_boost(df, top_k=10, stickiness_boost=0.18, prev_rank_col=None, hr_col='hr_probability'):
    stick = df[hr_col].copy()
    if prev_rank_col and prev_rank_col in df.columns:
        prev_rank = df[prev_rank_col].rank(method='min', ascending=False)
        stick = stick + stickiness_boost * (prev_rank <= top_k)
    else:
        stick.iloc[:top_k] += stickiness_boost
    return stick

def auto_feature_crosses(X, max_cross=24, template_cols=None):
    cross_names = []
    if template_cols is not None:
        for name in template_cols:
            f1, f2 = name.split('*')
            X[name] = X[f1] * X[f2]
            cross_names.append(name)
        X = X.copy()
        return X, cross_names
    means = X.mean()
    var_scores = {}
    cols = list(X.columns)
    for i, f1 in enumerate(cols):
        for j, f2 in enumerate(cols):
            if i >= j: continue
            cross = X[f1] * X[f2]
            var_scores[(f1, f2)] = cross.var()
    top_pairs = sorted(var_scores.items(), key=lambda kv: -kv[1])[:max_cross]
    for (f1, f2), _ in top_pairs:
        name = f"{f1}*{f2}"
        X[name] = X[f1] * X[f2]
        cross_names.append(name)
    X = X.copy()
    return X, cross_names

def remove_outliers(
    X,
    y,
    method="iforest",
    contamination=0.012,
    n_estimators=100,
    max_samples='auto',
    n_neighbors=20,
    scale=True
):
    """
    Remove outliers from the dataset using Isolation Forest or Local Outlier Factor.

    Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        method (str): "iforest" or "lof".
        contamination (float): Proportion of outliers in the data.
        n_estimators (int): Number of trees for IsolationForest.
        max_samples (str or int): Number of samples for IsolationForest.
        n_neighbors (int): Number of neighbors for LOF.
        scale (bool): Whether to standardize features before fitting.

    Returns:
        X_filtered, y_filtered: Data with outliers removed.
    """
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    if method == "iforest":
        clf = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=42
        )
        mask = clf.fit_predict(X_scaled) == 1
    elif method == "lof":
        clf = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors
        )
        mask = clf.fit_predict(X_scaled) == 1
    else:
        raise ValueError("Unknown method: choose 'iforest' or 'lof'")

    return X[mask], y[mask]
def smooth_labels(y, smoothing=0.02):
    y = np.asarray(y)
    y_smooth = y.copy().astype(float)
    y_smooth[y == 1] = 1 - smoothing
    y_smooth[y == 0] = smoothing
    return y_smooth

# ---- APP START ----

event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv', 'parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type=['csv', 'parquet'], key='todaycsv')

if event_file is not None and today_file is not None:
    with st.spinner("Loading and prepping files..."):
        event_df = safe_read_cached(event_file)
        today_df = safe_read_cached(today_file)
        event_df = event_df.dropna(axis=1, how='all')
        today_df = today_df.dropna(axis=1, how='all')
        event_df = dedup_columns(event_df)
        today_df = dedup_columns(today_df)
        event_df = event_df.reset_index(drop=True)
        today_df = today_df.reset_index(drop=True)
        if find_duplicate_columns(event_df):
            st.error(f"Duplicate columns in event file")
            st.stop()
        if find_duplicate_columns(today_df):
            st.error(f"Duplicate columns in today file")
            st.stop()
        event_df = fix_types(event_df)
        today_df = fix_types(today_df)
        st.write(f"event_df shape: {event_df.shape}, today_df shape: {today_df.shape}")
        st.write(f"event_df memory usage (MB): {event_df.memory_usage(deep=True).sum() / 1024**2:.2f}")
        st.write(f"today_df memory usage (MB): {today_df.memory_usage(deep=True).sum() / 1024**2:.2f}")

    target_col = 'hr_outcome'
    if target_col not in event_df.columns:
        st.error("ERROR: No valid hr_outcome column found in event-level file.")
        st.stop()
    st.success("✅ 'hr_outcome' column found in event-level data.")

    # ---- Feature Filtering ----
    feature_cols = sorted(list(set(get_valid_feature_cols(event_df)) & set(get_valid_feature_cols(today_df))))
    st.write(f"Feature count before filtering: {len(feature_cols)}")
    X = clean_X(event_df[feature_cols])
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)
    feature_debug(X)

    nan_thresh = 0.3
    nan_pct = X.isna().mean()
    drop_cols = nan_pct[nan_pct > nan_thresh].index.tolist()
    if drop_cols:
        st.warning(f"Dropping {len(drop_cols)} features with >30% NaNs: {drop_cols[:20]}")
        X = X.drop(columns=drop_cols)
        X_today = X_today.drop(columns=drop_cols, errors='ignore')

    nzv_cols = X.loc[:, X.nunique() <= 2].columns.tolist()
    if nzv_cols:
        st.warning(f"Dropping {len(nzv_cols)} near-constant features.")
        X = X.drop(columns=nzv_cols)
        X_today = X_today.drop(columns=nzv_cols, errors='ignore')

    corrs = X.corr().abs()
    upper = corrs.where(np.triu(np.ones(corrs.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.999)]
    if to_drop:
        st.warning(f"Dropping {len(to_drop)} highly correlated features.")
        X = X.drop(columns=to_drop)
        X_today = X_today.drop(columns=to_drop, errors='ignore')

    X = winsorize_clip(X)
    X_today = winsorize_clip(X_today)

    # --- Outlier removal ---
    st.write("🚦 Starting outlier removal...")
    y = event_df[target_col].astype(int)
    X, y = remove_outliers(X, y, method="iforest", contamination=0.012)
    X = X.reset_index(drop=True).copy()
    y = pd.Series(y).reset_index(drop=True)
    st.write(f"✅ Outlier removal complete. Rows remaining: {X.shape[0]}")

    # --- Fill missing values ---
    st.write("🩹 Filling missing values...")
    X = X.fillna(-1)
    X_today = X_today.fillna(-1)

    # --- Step 1: Initial model to rank base features ---
    st.write("🔬 Training initial XGBoost model to rank base features...")
    initial_model = XGBClassifier(n_estimators=100, max_depth=4, verbosity=0, random_state=42)
    initial_model.fit(X, y)
    base_importances = pd.Series(initial_model.feature_importances_, index=X.columns)

    # Deduplicate base_importances index if needed (usually not required here, but for safety)
    base_importances = base_importances.loc[~base_importances.index.duplicated()]

    top_base_features = base_importances.sort_values(ascending=False).head(30).index.tolist()
    st.write("🏆 Top base features selected:", top_base_features)

    # --- Step 2: Generate cross-features using interactions ---
    st.write("🔗 Generating cross-features...")

    @st.cache_data(show_spinner=False, max_entries=1)  # Cache to avoid recomputation
    def generate_cross_features(_X, top_features):
        try:
        # Memory check before processing
            gc.collect()
            mem_before = psutil.virtual_memory().available / (1024 ** 3)  # GB
        
            if mem_before < 2:  # Less than 2GB available
                st.warning("⚠️ Low memory detected. Using simplified feature crosses.")
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                X_cross = pd.DataFrame(
                    poly.fit_transform(_X[top_features].astype(np.float32)),  # Use float32 to save memory
                    columns=poly.get_feature_names_out(top_features)
                )
            else:
           # Full processing with memory monitoring
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                X_cross = pd.DataFrame(
                    poly.fit_transform(_X[top_features]),
                    columns=poly.get_feature_names_out(top_features)
                )
        
           # Deduplicate and clean
            X_cross = dedup_columns(X_cross)
        
        # Memory cleanup
            gc.collect()
            mem_after = psutil.virtual_memory().available / (1024 ** 3)
            st.write(f"🧠 Memory: {mem_before:.1f}GB → {mem_after:.1f}GB available after cross features")
        
            return X_cross
        
        except MemoryError:
            st.error("🚨 Out of memory during cross-feature generation. Falling back to base features.")
            return pd.DataFrame()  # Return empty as fallback
        except Exception as e:
            st.error(f"❌ Cross-feature generation failed: {str(e)}")
            return pd.DataFrame()

    # Generate with monitoring
    with st.spinner("Creating feature interactions (this may take a minute)..."):
        X_cross = generate_cross_features(X, top_base_features)
    
        if X_cross.empty:
            st.warning("Using base features only (cross-feature generation failed)")
            X_cross = pd.DataFrame(index=X.index)  # Empty but preserves row count
        else:
        # Validate output
            nan_inf_check(X_cross, "cross features")
            st.write(f"🔢 Cross-feature matrix shape: {X_cross.shape}")
            st.write("Sample cross features:", X_cross.iloc[:, :5].head(3))

    # --- Step 3: Combine and re-rank all features (base + cross) ---
    st.write("🧩 Combining base and cross features...")
    X_combined = pd.concat([X[top_base_features], X_cross], axis=1)

    # Deduplicate combined features too, just in case
    X_combined = dedup_columns(X_combined)

    st.write("📈 Fitting XGBoost to rank combined features...")
        xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_combined, y)
    coefs = pd.Series(xgb_model.feature_importances_, index=X_combined.columns)

    # Deduplicate coefficients index just in case
    coefs = coefs.loc[~coefs.index.duplicated()]

    top_combined_features = coefs.sort_values(ascending=False).head(200).index.tolist()  # <-- 200 here!
    st.write("🏁 Top combined features selected:", top_combined_features)

    # --- Final output ---
    st.write("🧼 Finalizing selected features...")

    X_selected = X_combined[top_combined_features].copy()

    # Align X_today to match columns and fill safely
    common_cols = X_selected.columns.intersection(X_today.columns)
    X_today_selected = X_today[common_cols].copy()

    # Reindex to ensure order matches X_selected, fill missing columns with -1
    X_today_selected = X_today_selected.reindex(columns=X_selected.columns, fill_value=-1)

    # Deduplicate final X_today_selected columns just in case
    X_today_selected = dedup_columns(X_today_selected)

    # Convert types FIRST before showing
    try:
        X_selected = X_selected.astype(np.float64)
        X_today_selected = X_today_selected.astype(np.float64)
        st.success("✅ Converted feature matrices to float64 for Streamlit compatibility")
    except Exception as e:
        st.error(f"❌ Conversion to float64 failed: {e}")

    # NOW safe to debug and display
    feature_debug(X_today_selected)
    st.dataframe(X_today_selected)

    # Final output confirmation
    st.write(f"✅ Final selected feature shape: {X_selected.shape}")
    st.write("🎯 Feature engineering and selection complete.")

    # --- Output preview ---
    st.write("📋 Preview of today's selected features:")
    st.dataframe(X_today_selected)

    # ========== OOS TEST =============
    OOS_ROWS = min(2000, len(X_selected) // 4)  # Use X_selected length here

    if len(X_selected) <= OOS_ROWS:
        st.warning(f"Dataset too small for OOS test. Using all {len(X_selected)} rows for training.")
        X_train = X_selected.copy()
        y_train = y.copy()
        X_oos = pd.DataFrame()
        y_oos = pd.Series()
    else:
        X_train = X_selected.iloc[:-OOS_ROWS].copy()
        y_train = y.iloc[:-OOS_ROWS].copy()
        X_oos = X_selected.iloc[-OOS_ROWS:].copy()
        y_oos = y.iloc[-OOS_ROWS:].copy()

    # ===== Sampling for Streamlit Cloud =====
    max_rows = 15000

    # Add defensive checks
    if 'X_train' not in locals() or X_train.empty:
        st.error("CRITICAL: X_train not properly initialized. Using selected features as fallback.")
        X_train = X_selected.copy()
        y_train = y.copy()
    
    if X_train.shape[0] > max_rows:
        st.warning(f"Training limited to {max_rows} rows for memory (full dataset was {X_train.shape[0]} rows).")
        X_train = X_train.iloc[:max_rows].copy()
        y_train = y_train.iloc[:max_rows].copy()

    # Final validation
    if X_train.empty or y_train.empty:
        st.error("FATAL: No training data available after sampling. Check your input data.")
        st.stop()

    st.write(f"✅ Final training data: {X_train.shape[0]} rows, {X_train.shape[1]} features")

    # ---- KFold Setup ----
    n_splits = 2
    n_repeats = 1
    st.write(f"Preparing KFold splits: X {X_train.shape}, y {y_train.shape}, X_today {X_today_selected.shape}")

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    val_fold_probas = np.zeros((len(y_train), 8))
    test_fold_probas = np.zeros((X_today_selected.shape[0], 8))
    scaler = StandardScaler()
    fold_times = []
    show_shap = st.checkbox("Show SHAP Feature Importance (slow, only for small datasets)", value=False)

    for fold, (tr_idx, va_idx) in enumerate(rskf.split(X_train, y_train)):
        t_fold_start = time.time()
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        sc = scaler.fit(X_tr)
        X_tr_scaled = sc.transform(X_tr)
        X_va_scaled = sc.transform(X_va)
        X_today_scaled = sc.transform(X_today_selected)
        # ... continue training & prediction ...
        # --- Optimized Tree Model Instantiations ---
        xgb_clf = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=1,
            verbosity=0
        )
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=7,
            num_leaves=31,
            learning_rate=0.07,
            subsample=0.8,
            feature_fraction=0.8,
            n_jobs=1
        )
        cat_clf = cb.CatBoostClassifier(
            iterations=150,
            depth=7,
            learning_rate=0.08,
            verbose=0,
            thread_count=1
        )
        rf_clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            max_features=0.7,
            min_samples_leaf=2,
            n_jobs=1
        )
        gb_clf = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8
        )
        lr_clf = LogisticRegression(
            max_iter=600,
            solver='lbfgs',
            n_jobs=1
        )

        xgb_clf.fit(X_tr_scaled, y_tr)
        lgb_clf.fit(X_tr_scaled, y_tr)
        cat_clf.fit(X_tr_scaled, y_tr)
        gb_clf.fit(X_tr_scaled, y_tr)
        rf_clf.fit(X_tr_scaled, y_tr)
        lr_clf.fit(X_tr_scaled, y_tr)

        val_fold_probas[va_idx, 0] = xgb_clf.predict_proba(X_va_scaled)[:, 1]
        val_fold_probas[va_idx, 1] = lgb_clf.predict_proba(X_va_scaled)[:, 1]
        val_fold_probas[va_idx, 2] = cat_clf.predict_proba(X_va_scaled)[:, 1]
        val_fold_probas[va_idx, 3] = gb_clf.predict_proba(X_va_scaled)[:, 1]
        val_fold_probas[va_idx, 4] = rf_clf.predict_proba(X_va_scaled)[:, 1]
        val_fold_probas[va_idx, 5] = lr_clf.predict_proba(X_va_scaled)[:, 1]
        val_fold_probas[va_idx, 6] = rf_clf.predict_proba(X_va_scaled)[:, 1]
        val_fold_probas[va_idx, 7] = lr_clf.predict_proba(X_va_scaled)[:, 1]

        test_fold_probas[:, 0] += xgb_clf.predict_proba(X_today_scaled)[:, 1] / (n_splits * n_repeats)
        test_fold_probas[:, 1] += lgb_clf.predict_proba(X_today_scaled)[:, 1] / (n_splits * n_repeats)
        test_fold_probas[:, 2] += cat_clf.predict_proba(X_today_scaled)[:, 1] / (n_splits * n_repeats)
        test_fold_probas[:, 3] += gb_clf.predict_proba(X_today_scaled)[:, 1] / (n_splits * n_repeats)
        test_fold_probas[:, 4] += rf_clf.predict_proba(X_today_scaled)[:, 1] / (n_splits * n_repeats)
        test_fold_probas[:, 5] += lr_clf.predict_proba(X_today_scaled)[:, 1] / (n_splits * n_repeats)
        test_fold_probas[:, 6] += rf_clf.predict_proba(X_today_scaled)[:, 1] / (n_splits * n_repeats)
        test_fold_probas[:, 7] += lr_clf.predict_proba(X_today_scaled)[:, 1] / (n_splits * n_repeats)

        if fold == 0 and show_shap:
            with st.spinner("Computing SHAP values (this can be slow)..."):
                explainer = shap.TreeExplainer(xgb_clf)
                shap_values = explainer.shap_values(X_va_scaled)
                st.write("Top SHAP Features (XGB, validation set):")
                shap.summary_plot(shap_values, pd.DataFrame(X_va_scaled, columns=X_tr.columns), show=False)
                st.pyplot(bbox_inches='tight')
                plt.clf()

        fold_time = time.time() - t_fold_start
        fold_times.append(fold_time)
        avg_time = np.mean(fold_times)
        est_time_left = avg_time * ((n_splits * n_repeats) - (fold + 1))
        st.write(f"Fold {fold + 1} finished in {timedelta(seconds=int(fold_time))}. Est. {timedelta(seconds=int(est_time_left))} left.")

    # Bagged predictions
    y_val_bag = val_fold_probas.mean(axis=1)
    y_today_bag = test_fold_probas.mean(axis=1)
    
    # ====== OOS TEST =======
    with st.spinner("🔍 Running Out-Of-Sample (OOS) test on last 2,000 rows..."):
        scaler_oos = StandardScaler()
        X_oos_train_scaled = scaler_oos.fit_transform(X_train)
        X_oos_scaled = scaler_oos.transform(X_oos)
        tree_models = [
            xgb.XGBClassifier(n_estimators=120, max_depth=7, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=0),
            lgb.LGBMClassifier(n_estimators=120, max_depth=7, learning_rate=0.08, n_jobs=1),
            cb.CatBoostClassifier(iterations=120, depth=7, learning_rate=0.08, verbose=0, thread_count=1),
            GradientBoostingClassifier(n_estimators=120, max_depth=7, learning_rate=0.08)
        ]
        hard_models = [
            RandomForestClassifier(n_estimators=120, max_depth=8, n_jobs=1),
            LogisticRegression(max_iter=600, solver='lbfgs', n_jobs=1)
        ]
        oos_preds = []
        for model in tree_models:
            model.fit(X_oos_train_scaled, y_train)
            oos_preds.append(model.predict_proba(X_oos_scaled)[:, 1])
        for model in hard_models:
            model.fit(X_oos_train_scaled, y_train)
            oos_preds.append(model.predict_proba(X_oos_scaled)[:, 1])
        oos_probs = np.mean(np.column_stack(oos_preds), axis=1)
        oos_auc = roc_auc_score(y_oos, oos_probs)
        oos_logloss = log_loss(y_oos, oos_probs)
        st.success(f"OOS AUC: {oos_auc:.4f} | OOS LogLoss: {oos_logloss:.4f}")

    # ==== OOS: Calibrated Model Performance Display ====
    st.markdown("### 📊 OOS Calibrated Model Performance (BetaCalibration, Isotonic, Blend)")

    oos_val_bag = np.mean(np.column_stack(oos_preds), axis=1)
    oos_bc = BetaCalibration(parameters="abm")
    oos_bc.fit(y_val_bag.reshape(-1,1), y_train)
    oos_pred_beta = oos_bc.predict(oos_val_bag.reshape(-1,1))
    oos_ir = IsotonicRegression(out_of_bounds="clip")
    oos_pred_iso = oos_ir.fit(y_val_bag, y_train).transform(oos_val_bag)
    oos_pred_blend = 0.5 * oos_pred_beta + 0.5 * oos_pred_iso

    oos_auc_beta = roc_auc_score(y_oos, oos_pred_beta)
    oos_logloss_beta = log_loss(y_oos, oos_pred_beta)
    oos_auc_iso = roc_auc_score(y_oos, oos_pred_iso)
    oos_logloss_iso = log_loss(y_oos, oos_pred_iso)
    oos_auc_blend = roc_auc_score(y_oos, oos_pred_blend)
    oos_logloss_blend = log_loss(y_oos, oos_pred_blend)

    st.write(f"**BetaCalibration:**   AUC = {oos_auc_beta:.4f}   |   LogLoss = {oos_logloss_beta:.4f}")
    st.write(f"**IsotonicRegression:**   AUC = {oos_auc_iso:.4f}   |   LogLoss = {oos_logloss_iso:.4f}")
    st.write(f"**Blended:**   AUC = {oos_auc_blend:.4f}   |   LogLoss = {oos_logloss_blend:.4f}")
    # ===== Calibration =====
    st.write("Calibrating probabilities (BetaCalibration & Isotonic & Blend)...")
    bc = BetaCalibration(parameters="abm")
    bc.fit(y_val_bag.reshape(-1,1), y_train)
    y_val_beta = bc.predict(y_val_bag.reshape(-1,1))
    y_today_beta = bc.predict(y_today_bag.reshape(-1,1))
    ir = IsotonicRegression(out_of_bounds="clip")
    y_val_iso = ir.fit_transform(y_val_bag, y_train)
    y_today_iso = ir.transform(y_today_bag)
    y_val_blend = 0.5 * y_val_beta + 0.5 * y_val_iso
    y_today_blend = 0.5 * y_today_beta + 0.5 * y_today_iso

    # ---- ADD WEATHER RATINGS ----
    today_df = today_df.copy()
    ratings_df = today_df.apply(rate_weather, axis=1)
    for col in ratings_df.columns:
        today_df[col] = ratings_df[col]

    def build_leaderboard(df, hr_probs, label="calibrated_hr_probability"):
        df = df.copy()
        df[label] = hr_probs
        df = df.sort_values(label, ascending=False).reset_index(drop=True)
        df['hr_base_rank'] = df[label].rank(method='min', ascending=False)
        if any([k in df.columns for k in ["wind_mph", "temp", "humidity", "park_hr_rate"]]):
            df['overlay_multiplier'] = df.apply(overlay_multiplier, axis=1)
            df['final_hr_probability'] = (df[label] * df['overlay_multiplier']).clip(0, 1)
            sort_col = "final_hr_probability"
        else:
            df['final_hr_probability'] = df[label]
            sort_col = "final_hr_probability"
        leaderboard_cols = []
        for c in ["player_name", "team_code", "time"]:
            if c in df.columns: leaderboard_cols.append(c)
        leaderboard_cols += [
            label, "final_hr_probability",
            "temp", "temp_rating",
            "humidity", "humidity_rating",
            "wind_mph", "wind_rating",
            "wind_dir_string", "condition", "condition_rating"
        ]
        if 'overlay_multiplier' in df.columns:
            leaderboard_cols.append('overlay_multiplier')
        leaderboard = df[leaderboard_cols].sort_values(sort_col, ascending=False).reset_index(drop=True)
        leaderboard[label] = leaderboard[label].round(4)
        leaderboard["final_hr_probability"] = leaderboard["final_hr_probability"].round(4)
        if 'overlay_multiplier' in leaderboard.columns:
            leaderboard['overlay_multiplier'] = leaderboard['overlay_multiplier'].round(3)
        return leaderboard, sort_col

    leaderboard_blend, sort_col_blend = build_leaderboard(today_df, y_today_blend, "hr_probability_blend")

    # Download and display for Blend leaderboard only
    top_n = 30
    st.markdown(f"### 🏆 **Top {top_n} HR Leaderboard (Blended BetaCal/Isotonic)**")
    leaderboard_top_blend = leaderboard_blend.head(top_n)
    st.dataframe(leaderboard_top_blend, use_container_width=True)
    st.download_button(
        f"⬇️ Download Top {top_n} Leaderboard (Blend) CSV",
        data=leaderboard_top_blend.to_csv(index=False),
        file_name=f"top{top_n}_leaderboard_blend.csv"
    )

    # Download full predictions
    st.download_button(
        "⬇️ Download Full Prediction CSV (Blend)",
        data=leaderboard_blend.to_csv(index=False),
        file_name="today_hr_predictions_full_blend.csv"
    )

    # Leaderboard plots
    if "player_name" in leaderboard_top_blend.columns:
        st.subheader(f"📊 HR Probability Distribution (Top 30, Blend)")
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(leaderboard_top_blend["player_name"].astype(str), leaderboard_top_blend[sort_col_blend], color='royalblue')
        ax.invert_yaxis()
        ax.set_xlabel('Predicted HR Probability')
        ax.set_ylabel('Player')
        st.pyplot(fig)

    # Drift diagnostics
    drifted = drift_check(X, X_today, n=6)
    if drifted:
        st.markdown("#### ⚡ **Feature Drift Diagnostics**")
        st.write("These features have unusual mean/std changes between training and today, check if input context shifted:", drifted)

    # Extra: Show prediction histogram (full today set, blend only)
    st.subheader(f"Prediction Probability Distribution (all predictions, Blend)")
    plt.figure(figsize=(8, 3))
    plt.hist(leaderboard_blend[sort_col_blend], bins=30, color='orange', alpha=0.7)
    plt.xlabel("Final HR Probability")
    plt.ylabel("Count")
    st.pyplot(plt.gcf())
    plt.close()

    # Memory cleanup
    del X, X_today, y, val_fold_probas, test_fold_probas, y_train, y_oos, X_train, X_oos
    gc.collect()

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
