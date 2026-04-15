from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json, os, subprocess, tempfile, traceback
from io import StringIO
from datetime import datetime
 
# --- Data & ML Libraries ---
import pandas as pd
import numpy as np
 
# Sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_squared_error, r2_score, silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
 
# Visualisation (saves to file, no display needed in server mode)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for server
import matplotlib.pyplot as plt
import seaborn as sns
 
# Stats
from scipy import stats
 
# LLM
from langchain_community.llms import Ollama
 
app = FastAPI(
    title="Deepthi's Data Science Agent",
    description="Autonomous Data Analysis, ML, Cleaning, Pipelines & Reporting",
    version="2.0"
)
 
# Allow cross-origin requests so dashboard can call this agent directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# --- LLM Setup ---
# Uses mistral via Ollama. Change model="tinyllama" here if RAM is tight.
llm = Ollama(model="mistral")
 
# =============================================================================
# SHARED API CONTRACT — matches shared/api_contracts/contract.json exactly
# =============================================================================
class TaskInput(BaseModel):
    task_id: str
    task_type: str = "data_analysis"
    description: str
    context: str = ""         # CSV data as string, or JSON config, or file path
    priority: str = "medium"
 
class TaskOutput(BaseModel):
    task_id: str
    status: str               # success | error | partial
    result: str               # main output
    summary: str
    next_agent: Optional[str] = None
    logs: list
 
# =============================================================================
# INTERNAL MODELS — used by sub-features, not exposed to contract
# =============================================================================
class CleaningConfig(BaseModel):
    strategy_missing: str = "mean"    # mean | median | mode | drop | ffill
    remove_duplicates: bool = True
    outlier_method: str = "iqr"       # iqr | zscore | none
    zscore_threshold: float = 3.0
 
# =============================================================================
# DUMMY DATA GENERATOR
# When user asks to "build", "create", "generate" data — we make realistic data
# =============================================================================
DUMMY_DATASETS = {
    "sales": lambda n: pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
        "product": np.random.choice(["Laptop","Phone","Tablet","Watch","Headphones"], n),
        "region": np.random.choice(["North","South","East","West"], n),
        "units_sold": np.random.randint(1, 200, n),
        "unit_price": np.round(np.random.uniform(50, 2000, n), 2),
        "revenue": None,  # will be computed
        "customer_age": np.random.randint(18, 70, n),
        "rating": np.round(np.random.uniform(1, 5, n), 1),
    }),
    "hr": lambda n: pd.DataFrame({
        "employee_id": range(1001, 1001 + n),
        "department": np.random.choice(["Engineering","Marketing","Sales","HR","Finance"], n),
        "salary": np.random.randint(30000, 150000, n),
        "years_experience": np.random.randint(0, 25, n),
        "performance_score": np.round(np.random.uniform(1, 5, n), 1),
        "left_company": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        "training_hours": np.random.randint(0, 100, n),
        "satisfaction": np.round(np.random.uniform(1, 10, n), 1),
    }),
    "ecommerce": lambda n: pd.DataFrame({
        "order_id": range(10001, 10001 + n),
        "user_id": np.random.randint(1, 500, n),
        "product_category": np.random.choice(["Electronics","Clothing","Books","Home","Sports"], n),
        "order_value": np.round(np.random.exponential(80, n) + 10, 2),
        "shipping_days": np.random.randint(1, 10, n),
        "return_flag": np.random.choice([0, 1], n, p=[0.88, 0.12]),
        "review_score": np.random.choice([1,2,3,4,5], n, p=[0.05,0.1,0.2,0.35,0.3]),
        "discount_pct": np.random.choice([0, 5, 10, 15, 20, 25], n),
    }),
    "health": lambda n: pd.DataFrame({
        "patient_id": range(1, n + 1),
        "age": np.random.randint(18, 90, n),
        "bmi": np.round(np.random.normal(26, 5, n), 1),
        "blood_pressure": np.random.randint(80, 180, n),
        "cholesterol": np.random.randint(150, 300, n),
        "glucose": np.random.randint(70, 200, n),
        "smoker": np.random.choice([0, 1], n, p=[0.75, 0.25]),
        "diabetic": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "heart_disease": np.random.choice([0, 1], n, p=[0.9, 0.1]),
    }),
    "generic": lambda n: pd.DataFrame({
        "id": range(1, n + 1),
        "feature_A": np.random.normal(50, 15, n).round(2),
        "feature_B": np.random.normal(100, 30, n).round(2),
        "feature_C": np.random.choice(["cat1","cat2","cat3"], n),
        "feature_D": np.random.randint(0, 100, n),
        "target": np.random.choice([0, 1], n),
    })
}
 
def get_dummy_dataset(description: str, n: int = 200) -> pd.DataFrame:
    """Pick the most relevant dummy dataset based on task description keywords."""
    desc = description.lower()
    if any(w in desc for w in ["sale","revenue","product","retail","shop","price"]):
        df = DUMMY_DATASETS["sales"](n)
        df["revenue"] = (df["units_sold"] * df["unit_price"]).round(2)
    elif any(w in desc for w in ["employee","hr","attrition","salary","staff","hiring"]):
        df = DUMMY_DATASETS["hr"](n)
    elif any(w in desc for w in ["ecommerce","order","cart","customer","shop","return"]):
        df = DUMMY_DATASETS["ecommerce"](n)
    elif any(w in desc for w in ["health","patient","medical","disease","clinical","hospital"]):
        df = DUMMY_DATASETS["health"](n)
    else:
        df = DUMMY_DATASETS["generic"](n)
    return df
 
# =============================================================================
# FEATURE DETECTION — what does the user actually want?
# =============================================================================
def detect_task_intent(description: str) -> str:
    """Route to the right analysis mode based on natural language description."""
    desc = description.lower()
    if any(w in desc for w in ["clean","fix","missing","duplicate","outlier","preprocess","impute","format"]):
        return "clean"
    if any(w in desc for w in ["pipeline","etl","transform","workflow","process","ingest","extract"]):
        return "pipeline"
    if any(w in desc for w in ["predict","classify","model","train","ml","machine learning","forecast","regression","classification"]):
        return "ml"
    if any(w in desc for w in ["cluster","segment","group","kmeans","unsupervised"]):
        return "cluster"
    if any(w in desc for w in ["visuali","plot","chart","graph","dashboard","visual"]):
        return "visualize"
    if any(w in desc for w in ["statistic","distribution","correlation","hypothesis","test","anova","t-test","chi"]):
        return "statistics"
    if any(w in desc for w in ["feature","importance","selection","pca","dimension","reduce"]):
        return "feature_engineering"
    if any(w in desc for w in ["report","summary","insight","overview","analyse","analyze","eda","explore"]):
        return "eda"
    if any(w in desc for w in ["build","create","generate","make","dummy","sample","synthetic","fake data"]):
        return "generate"
    return "eda"  # default: exploratory analysis
 
# =============================================================================
# CORE ANALYSIS FUNCTIONS
# =============================================================================
 
def run_eda(df: pd.DataFrame, logs: list) -> str:
    """Full exploratory data analysis."""
    logs.append("Running EDA: shape, types, missing values, distributions")
    report_parts = []
 
    # Basic info
    report_parts.append(f"DATASET OVERVIEW\n{'='*40}")
    report_parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    report_parts.append(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
 
    # Data types
    type_summary = df.dtypes.value_counts().to_dict()
    report_parts.append(f"Column types: {type_summary}")
 
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    if missing.sum() > 0:
        missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
        missing_df = missing_df[missing_df["Missing Count"] > 0]
        report_parts.append(f"\nMISSING VALUES\n{missing_df.to_string()}")
    else:
        report_parts.append("\nMissing Values: None ✓")
 
    # Duplicates
    dups = df.duplicated().sum()
    report_parts.append(f"Duplicate rows: {dups}")
 
    # Numeric stats
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        stats_df = df[numeric_cols].describe().round(3)
        report_parts.append(f"\nNUMERIC STATISTICS\n{stats_df.to_string()}")
 
    # Correlations (top pairs)
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        # Find highest correlations (excluding self)
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i,j]))
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_corr = corr_pairs[:5]
        report_parts.append("\nTOP CORRELATIONS:")
        for c1, c2, val in top_corr:
            report_parts.append(f"  {c1} ↔ {c2}: {val:.3f}")
 
    # Categorical columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        report_parts.append("\nCATEGORICAL COLUMNS:")
        for col in cat_cols[:5]:
            vc = df[col].value_counts()
            report_parts.append(f"  {col}: {vc.index[:3].tolist()} (top 3 of {df[col].nunique()} unique)")
 
    # Outlier detection (IQR method)
    outlier_summary = []
    for col in numeric_cols[:6]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        n_outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        if n_outliers > 0:
            outlier_summary.append(f"  {col}: {n_outliers} outliers ({n_outliers/len(df)*100:.1f}%)")
    if outlier_summary:
        report_parts.append("\nOUTLIERS DETECTED (IQR method):")
        report_parts.extend(outlier_summary)
 
    raw_report = "\n".join(report_parts)
 
    # Ask LLM to generate natural-language insights from the stats
    logs.append("Generating AI insights from statistics")
    insight_prompt = f"""
You are a senior data analyst. Based on this dataset analysis, write 3 clear business insights.
Be specific — use the actual numbers. Mention patterns, risks, and opportunities.
 
Raw stats:
{raw_report[:3000]}
 
Write exactly 3 insights in plain English, numbered 1, 2, 3. Keep each insight to 2-3 sentences.
"""
    insights = llm(insight_prompt)
    return raw_report + f"\n\n{'='*40}\nAI-GENERATED INSIGHTS\n{'='*40}\n{insights}"
 
 
def run_cleaning(df: pd.DataFrame, config: CleaningConfig, logs: list) -> tuple:
    """Full data cleaning pipeline. Returns (cleaned_df, cleaning_report)."""
    report = []
    original_shape = df.shape
 
    # 1. Remove duplicates
    if config.remove_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        report.append(f"Removed {removed} duplicate rows")
        logs.append(f"Deduplication: removed {removed} rows")
 
    # 2. Handle missing values
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
 
    strategy = config.strategy_missing
    logs.append(f"Imputing missing values with strategy: {strategy}")
 
    if strategy == "drop":
        before = len(df)
        df = df.dropna()
        report.append(f"Dropped rows with missing values: {before - len(df)} removed")
    elif strategy == "ffill":
        df = df.ffill()
        df = df.bfill()
        report.append("Applied forward-fill for missing values (time series safe)")
    else:
        # Numeric imputation
        if numeric_cols:
            num_imputer = SimpleImputer(strategy=strategy if strategy in ["mean","median"] else "mean")
            df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
            report.append(f"Numeric columns imputed with '{strategy}'")
        # Categorical imputation — always mode
        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
            report.append("Categorical columns imputed with most frequent value")
 
    # 3. Outlier handling
    if config.outlier_method == "iqr":
        clipped = 0
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            outliers_here = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower, upper)
            clipped += outliers_here
        report.append(f"Clipped {clipped} outlier values using IQR method")
        logs.append(f"Outlier clipping: {clipped} values adjusted")
    elif config.outlier_method == "zscore":
        clipped = 0
        for col in numeric_cols:
            z = np.abs(stats.zscore(df[col].dropna()))
            n_out = (z > config.zscore_threshold).sum()
            mean, std = df[col].mean(), df[col].std()
            df[col] = df[col].clip(
                mean - config.zscore_threshold * std,
                mean + config.zscore_threshold * std
            )
            clipped += n_out
        report.append(f"Clipped {clipped} outlier values using Z-score (threshold={config.zscore_threshold})")
 
    # 4. Type inference — fix columns that look numeric but are string
    for col in cat_cols:
        try:
            df[col] = pd.to_numeric(df[col])
            report.append(f"Auto-converted '{col}' from string to numeric")
        except (ValueError, TypeError):
            pass
 
    final_shape = df.shape
    report.insert(0, f"Original shape: {original_shape} → Cleaned shape: {final_shape}")
 
    cleaning_report = "\n".join(report)
 
    # LLM summary
    llm_summary = llm(f"""
You are a data engineer. Summarize these data cleaning steps in 2 sentences for a business audience.
Don't use technical jargon. Steps: {cleaning_report}
""")
    return df, cleaning_report + f"\n\nSUMMARY: {llm_summary}"
 
 
def run_ml(df: pd.DataFrame, description: str, logs: list) -> str:
    """Auto-ML: detect target, choose model, train, evaluate."""
    logs.append("Starting Auto-ML pipeline")
    desc = description.lower()
 
    # Detect target column
    target_col = None
    possible_targets = ["target","label","class","churn","left_company","diabetic","heart_disease",
                        "return_flag","outcome","y","result","status"]
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]  # fallback: last column
 
    logs.append(f"Target column identified: {target_col}")
 
    # Prepare features
    X = df.drop(columns=[target_col])
    y = df[target_col]
 
    # Encode categoricals
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_encoded = X_encoded.select_dtypes(include=np.number).fillna(0)
 
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
 
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
    # Determine task type
    n_unique = y.nunique()
    is_classification = n_unique <= 20 or y.dtype == object
 
    results = []
 
    if is_classification:
        logs.append("Task type: Classification")
        # Encode target if needed
        if y.dtype == object:
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
 
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        }
        best_score, best_name, best_model = 0, "", None
        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            cv_scores = cross_val_score(model, X_scaled, y if y.dtype != object else le.transform(y),
                                        cv=3, scoring="accuracy")
            results.append(f"{name}: Test Accuracy={score:.4f}, CV Mean={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            if score > best_score:
                best_score, best_name, best_model = score, name, model
        logs.append(f"Best model: {best_name}")
 
        # Feature importance (if RF or GB)
        feat_importance = ""
        if hasattr(best_model, "feature_importances_"):
            fi = pd.Series(best_model.feature_importances_, index=X_encoded.columns)
            top5 = fi.nlargest(5)
            feat_importance = "\nTOP 5 FEATURE IMPORTANCES:\n" + "\n".join(
                [f"  {k}: {v:.4f}" for k, v in top5.items()])
 
        report = f"""AUTO-ML CLASSIFICATION REPORT
{'='*40}
Target: {target_col} | Classes: {n_unique} | Training rows: {len(X_train)}
 
MODEL COMPARISON:
{chr(10).join(results)}
 
BEST MODEL: {best_name} (Accuracy: {best_score:.4f})
{feat_importance}
"""
    else:
        logs.append("Task type: Regression")
        models = {
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
        }
        best_r2, best_name = -999, ""
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            results.append(f"{name}: R²={r2:.4f}, RMSE={rmse:.4f}")
            if r2 > best_r2:
                best_r2, best_name = r2, name
 
        report = f"""AUTO-ML REGRESSION REPORT
{'='*40}
Target: {target_col} | Training rows: {len(X_train)}
 
MODEL COMPARISON:
{chr(10).join(results)}
 
BEST MODEL: {best_name} (R²: {best_r2:.4f})
"""
 
    # LLM interpretation
    interpretation = llm(f"""
You are a data science expert. Interpret these ML results for a business team in 3 sentences.
Explain what the accuracy/R² means in plain English and what the team should do next.
Results: {report[:1500]}
""")
    return report + f"\nINTERPRETATION:\n{interpretation}"
 
 
def run_clustering(df: pd.DataFrame, description: str, logs: list) -> str:
    """Clustering analysis with auto k selection."""
    logs.append("Running clustering analysis")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        return "No numeric columns found for clustering."
 
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    # Find optimal k using silhouette score (k=2 to 8)
    best_k, best_sil = 3, -1
    sil_scores = {}
    for k in range(2, min(9, len(df)//10 + 2)):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        sil_scores[k] = round(sil, 4)
        if sil > best_sil:
            best_sil, best_k = sil, k
 
    logs.append(f"Optimal clusters: {best_k} (silhouette={best_sil:.4f})")
 
    # Final clustering
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)
 
    # Cluster profiles
    cluster_profiles = df.groupby("cluster")[numeric_cols].mean().round(3)
 
    report = f"""CLUSTERING ANALYSIS REPORT
{'='*40}
Silhouette scores by k: {sil_scores}
Optimal k: {best_k} clusters (score={best_sil:.4f})
 
CLUSTER PROFILES (mean values):
{cluster_profiles.to_string()}
 
Cluster sizes:
{df['cluster'].value_counts().sort_index().to_string()}
"""
 
    # LLM name the clusters
    naming_prompt = f"""
You are a business analyst. Given these cluster profiles from a dataset, give each cluster a short descriptive name (3-4 words).
Profiles: {cluster_profiles.to_string()}
Format: Cluster 0: [name], Cluster 1: [name], etc.
"""
    cluster_names = llm(naming_prompt)
    return report + f"\nCLUSTER LABELS (AI-named):\n{cluster_names}"
 
 
def run_statistics(df: pd.DataFrame, description: str, logs: list) -> str:
    """Statistical tests and distribution analysis."""
    logs.append("Running statistical analysis")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    results = []
 
    for col in numeric_cols[:8]:
        col_data = df[col].dropna()
        # Normality test
        if len(col_data) >= 8:
            stat, p = stats.shapiro(col_data[:5000])  # Shapiro max 5000
            normal = "Normal" if p > 0.05 else "Non-normal"
            skew = col_data.skew()
            kurt = col_data.kurtosis()
            results.append(f"{col}: {normal} (p={p:.4f}), Skew={skew:.3f}, Kurtosis={kurt:.3f}")
 
    # Correlation matrix summary
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        strong_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                val = corr.iloc[i,j]
                if abs(val) > 0.5:
                    strong_pairs.append(f"  {corr.columns[i]} & {corr.columns[j]}: r={val:.3f}")
 
    stat_summary = "\n".join(results)
    corr_summary = "\n".join(strong_pairs) if strong_pairs else "No strong correlations (r > 0.5)"
 
    report = f"""STATISTICAL ANALYSIS REPORT
{'='*40}
DISTRIBUTION TESTS (Shapiro-Wilk):
{stat_summary}
 
STRONG CORRELATIONS (|r| > 0.5):
{corr_summary}
"""
 
    interpretation = llm(f"""
You are a statistician. Explain these results to a non-technical business audience in 3 sentences.
Focus on what the distributions and correlations mean for decision-making.
Results: {report[:2000]}
""")
    return report + f"\nINSIGHT:\n{interpretation}"
 
 
def run_feature_engineering(df: pd.DataFrame, description: str, logs: list) -> str:
    """Feature importance, PCA, selection."""
    logs.append("Running feature engineering analysis")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        return "Need at least 2 numeric columns for feature engineering."
 
    # PCA
    X = df[numeric_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X_scaled)
    explained = np.cumsum(pca.explained_variance_ratio_)
    n_for_95 = int(np.argmax(explained >= 0.95)) + 1
 
    # Variance of each feature
    variance_info = pd.Series(dict(zip(numeric_cols, X.var().round(3)))).sort_values(ascending=False)
 
    report = f"""FEATURE ENGINEERING REPORT
{'='*40}
PCA ANALYSIS:
  Total features: {len(numeric_cols)}
  Features needed for 95% variance: {n_for_95}
  PCA explained variance per component: {[round(v,3) for v in pca.explained_variance_ratio_[:6]]}
 
FEATURE VARIANCE (higher = more informative):
{variance_info.head(10).to_string()}
"""
 
    guidance = llm(f"""
You are a senior ML engineer. Based on this feature analysis, give 3 concrete recommendations.
Which features to keep, drop, or engineer? Be specific.
Analysis: {report[:2000]}
""")
    return report + f"\nRECOMMENDATIONS:\n{guidance}"
 
 
def run_pipeline(df: pd.DataFrame, description: str, logs: list) -> tuple:
    """Data pipeline: clean → feature engineer → prepare for ML."""
    logs.append("Building data pipeline")
    steps_log = []
 
    # Step 1: Clean
    config = CleaningConfig()
    df, clean_report = run_cleaning(df, config, logs)
    steps_log.append(f"Step 1 - Cleaning:\n{clean_report}")
 
    # Step 2: Encode categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        if df[col].nunique() <= 20:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = "LabelEncoded"
        else:
            df = df.drop(columns=[col])  # too many categories — drop
            encoders[col] = "Dropped (cardinality too high)"
    steps_log.append(f"Step 2 - Encoding: {encoders}")
    logs.append("Categorical encoding complete")
 
    # Step 3: Scale numerics
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        steps_log.append(f"Step 3 - Scaling: MinMax scaling applied to {len(numeric_cols)} columns")
        logs.append("Feature scaling complete")
 
    # Step 4: Report
    report = f"""DATA PIPELINE REPORT
{'='*40}
Pipeline: Ingest → Clean → Encode → Scale → Ready for ML
 
{chr(10).join(steps_log)}
 
FINAL DATASET:
  Shape: {df.shape}
  All numeric: {df.select_dtypes(include=np.number).shape[1]} columns
  Ready for ML: Yes ✓
"""
 
    pipeline_code = llm(f"""
Write a short Python code snippet (15 lines max) that implements this pipeline as a reusable function.
Use sklearn Pipeline. Pipeline steps: {[k for k in encoders.items()]}
Make it clean and production-ready.
""")
    return df, report + f"\nREUSABLE PIPELINE CODE:\n{pipeline_code}"
 
 
def run_visualization(df: pd.DataFrame, description: str, logs: list) -> str:
    """Generate charts and return paths + description."""
    logs.append("Generating visualizations")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
 
    charts_created = []
    output_dir = "/tmp/agent_charts"
    os.makedirs(output_dir, exist_ok=True)
 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    # 1. Distribution plots
    if numeric_cols:
        fig, axes = plt.subplots(2, min(3, len(numeric_cols)), figsize=(15, 8))
        axes = np.array(axes).flatten()
        for i, col in enumerate(numeric_cols[:6]):
            axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
        plt.tight_layout()
        path = f"{output_dir}/distributions_{timestamp}.png"
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        charts_created.append(path)
        logs.append("Created distribution plots")
 
    # 2. Correlation heatmap
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Heatmap')
        plt.tight_layout()
        path = f"{output_dir}/correlation_{timestamp}.png"
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        charts_created.append(path)
        logs.append("Created correlation heatmap")
 
    # 3. Categorical counts
    if cat_cols:
        col = cat_cols[0]
        fig, ax = plt.subplots(figsize=(8, 5))
        df[col].value_counts().head(10).plot(kind='bar', ax=ax, color='coral', edgecolor='black')
        ax.set_title(f'{col} — Value Counts')
        ax.set_ylabel('Count')
        plt.tight_layout()
        path = f"{output_dir}/categorical_{timestamp}.png"
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        charts_created.append(path)
        logs.append("Created categorical bar chart")
 
    # 4. Boxplots for outlier visualization
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(12, 5))
        cols_to_plot = numeric_cols[:6]
        df_plot = df[cols_to_plot].copy()
        # Normalize for display
        df_norm = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min() + 1e-9)
        df_norm.boxplot(ax=ax)
        ax.set_title('Boxplots (Normalized) — Outlier Detection')
        plt.tight_layout()
        path = f"{output_dir}/boxplots_{timestamp}.png"
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        charts_created.append(path)
 
    chart_desc = llm(f"""
You are a data visualization expert. Describe what each of these {len(charts_created)} charts shows.
Charts generated: {[os.path.basename(p) for p in charts_created]}
Dataset columns: {df.columns.tolist()}
Write 1 sentence per chart explaining what to look for.
""")
 
    return f"""VISUALIZATIONS GENERATED
{'='*40}
Charts created: {len(charts_created)}
Saved to: {output_dir}/
 
Files:
{chr(10).join(['  - ' + os.path.basename(p) for p in charts_created])}
 
{chart_desc}
 
NOTE: In standalone mode charts are saved to /tmp/agent_charts/. 
When integrated with the platform dashboard, Ayeesha's agent will render them.
"""
 
# =============================================================================
# MAIN ROUTE — matches API contract exactly
# =============================================================================
@app.post("/run", response_model=TaskOutput)
def run_task(task: TaskInput):
    logs = [f"Task received: {task.task_id} | Priority: {task.priority}"]
    
    try:
        # -----------------------------------------------
        # 1. LOAD DATA
        # -----------------------------------------------
        df = None
        data_source = "none"
 
        if task.context and task.context.strip():
            # Try to parse as CSV string
            try:
                df = pd.read_csv(StringIO(task.context))
                data_source = "user_provided_csv"
                logs.append(f"Loaded user CSV: {df.shape}")
            except Exception:
                pass
 
            # Try JSON
            if df is None:
                try:
                    data = json.loads(task.context)
                    df = pd.DataFrame(data)
                    data_source = "user_provided_json"
                    logs.append(f"Loaded user JSON: {df.shape}")
                except Exception:
                    pass
 
        # No data provided — generate dummy
        if df is None:
            intent = detect_task_intent(task.description)
            if intent == "generate" or task.context == "" or df is None:
                df = get_dummy_dataset(task.description, n=250)
                data_source = "auto_generated_dummy"
                logs.append(f"Auto-generated dummy dataset ({df.shape}) based on task keywords")
 
        # Final fallback
        if df is None or df.empty:
            df = get_dummy_dataset("generic", n=200)
            data_source = "fallback_generic"
            logs.append("Using fallback generic dataset")
 
        # -----------------------------------------------
        # 2. DETECT INTENT AND ROUTE
        # -----------------------------------------------
        intent = detect_task_intent(task.description)
        logs.append(f"Detected task intent: {intent} | Data source: {data_source}")
 
        result = ""
        next_agent = None  # Data agent typically ends the pipeline
        # (set to "security" if code is generated, to "devops" for pipelines)
 
        # -----------------------------------------------
        # 3. EXECUTE THE RIGHT ANALYSIS
        # -----------------------------------------------
        if intent == "eda":
            result = run_eda(df, logs)
 
        elif intent == "clean":
            config = CleaningConfig()
            desc = task.description.lower()
            if "median" in desc: config.strategy_missing = "median"
            elif "mode" in desc: config.strategy_missing = "mode"
            elif "drop" in desc: config.strategy_missing = "drop"
            if "zscore" in desc or "z-score" in desc: config.outlier_method = "zscore"
            df_clean, cleaning_result = run_cleaning(df, config, logs)
            result = cleaning_result + f"\n\nCLEANED DATA PREVIEW (first 5 rows):\n{df_clean.head().to_string()}"
 
        elif intent == "ml":
            result = run_ml(df, task.description, logs)
 
        elif intent == "cluster":
            result = run_clustering(df, task.description, logs)
 
        elif intent == "visualize":
            result = run_visualization(df, task.description, logs)
 
        elif intent == "statistics":
            result = run_statistics(df, task.description, logs)
 
        elif intent == "feature_engineering":
            result = run_feature_engineering(df, task.description, logs)
 
        elif intent == "pipeline":
            df_processed, pipeline_result = run_pipeline(df, task.description, logs)
            result = pipeline_result
            next_agent = "devops"  # Processed data → deployment makes sense
 
        elif intent == "generate":
            # Just generate and profile
            eda_result = run_eda(df, logs)
            result = f"""GENERATED DATASET INFO
{'='*40}
Data source: Auto-generated ({data_source})
Shape: {df.shape}
 
SAMPLE DATA (first 5 rows):
{df.head().to_string()}
 
{eda_result}
"""
        else:
            result = run_eda(df, logs)
 
        # -----------------------------------------------
        # 4. FINAL SUMMARY VIA LLM
        # -----------------------------------------------
        summary = llm(f"""
In 2 sentences, summarize what was accomplished in this data task.
Task: {task.description}
Intent detected: {intent}
Data source: {data_source}
Be concise and professional.
""")
        logs.append("Task completed successfully")
 
        return TaskOutput(
            task_id=task.task_id,
            status="success",
            result=result,
            summary=summary,
            next_agent=next_agent,
            logs=logs
        )
 
    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"Agent error: {str(e)}\n\nTraceback:\n{tb}"
        logs.append(f"ERROR: {str(e)}")
        return TaskOutput(
            task_id=task.task_id,
            status="error",
            result=error_msg,
            summary=f"Task failed: {str(e)}",
            next_agent=None,
            logs=logs
        )
 
# =============================================================================
# HEALTH CHECK — used by Likitha's docker-compose health checks
# =============================================================================
@app.get("/health")
def health():
    return {"status": "ok", "agent": "deepthi_data_science", "port": 8001, "model": "mistral"}
 
@app.get("/")
def root():
    return {
        "agent": "Deepthi's Data Science & Analytics Agent",
        "version": "2.0",
        "port": 8001,
        "capabilities": [
            "EDA (Exploratory Data Analysis)",
            "Data Cleaning & Preprocessing",
            "Auto-ML (Classification & Regression)",
            "Clustering & Segmentation",
            "Statistical Analysis",
            "Feature Engineering & PCA",
            "Data Pipeline Building",
            "Visualization Generation",
            "Dummy/Synthetic Data Generation",
        ],
        "api_contract": "compatible with shared/api_contracts/contract.json",
        "docs": "/docs"
    }