import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    confusion_matrix,
    precision_recall_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Cheese Vegetarian Predictor", layout="wide", page_icon="🧀")

# Title and description
st.title("🧀 Cheese Vegetarian Predictor")
st.markdown("""
This app predicts whether a cheese is **vegetarian** or **non-vegetarian** based on its characteristics.

**How it works:**
- Upload a cheese dataset OR use the default dataset
- The model analyzes milk type, texture, aroma, country, and more
- Get predictions with detailed model performance metrics
""")

# Sidebar
st.sidebar.header("⚙️ Configuration")
use_default = st.sidebar.checkbox("Use default cheese dataset", value=True)
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Logistic Regression", "Decision Tree"]
)

# Load data
@st.cache_data
def load_data(use_default_data=True):
    if use_default_data:
        url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2024/2024-06-04/cheeses.csv"
        df = pd.read_csv(url)
    else:
        return None
    return df

@st.cache_data
def preprocess_data(df):
    """Preprocess the cheese dataset"""
    # Drop rows with missing target
    df = df.dropna(subset=["vegetarian"])
    
    # Drop irrelevant columns
    columns_to_drop = ["url", "cheese"]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
    
    # Simplify multi-valued categories
    multi_valued_cols = ["milk", "type", "texture", "aroma"]
    for col in multi_valued_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.split(",").str[0]
    
    # Convert target to binary
    df["vegetarian"] = df["vegetarian"].map({True: 1, False: 0})
    
    return df

# Main app logic
if use_default:
    with st.spinner("Loading data..."):
        df = load_data(use_default_data=True)
        
        if df is not None:
            st.success("✅ Data loaded successfully!")
            
            # Show dataset info
            with st.expander("📊 View Dataset"):
                st.write(f"**Dataset Shape:** {df.shape}")
                st.dataframe(df.head(10))
            
            # Preprocess
            df = preprocess_data(df)
            
            # Show target distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Target Distribution")
                fig1, ax1 = plt.subplots(figsize=(6,4))
                df["vegetarian"].value_counts().plot(kind='bar', ax=ax1, color=['#ff6b6b', '#4ecdc4'])
                ax1.set_xlabel("Vegetarian (0=No, 1=Yes)")
                ax1.set_ylabel("Count")
                ax1.set_title("Class Distribution")
                plt.xticks(rotation=0)
                st.pyplot(fig1)
            
            with col2:
                st.write("### Vegetarian Rate by Milk Type")
                fig2, ax2 = plt.subplots(figsize=(6,4))
                milk_rate = df.groupby("milk")["vegetarian"].mean().sort_values(ascending=False)
                milk_rate.plot(kind="bar", ax=ax2, color='#95e1d3')
                ax2.set_ylabel("Vegetarian Rate")
                ax2.set_title("Milk Type Analysis")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig2)
            
            # Prepare features and target
            X = df.drop(columns=["vegetarian"])
            y = df["vegetarian"]
            
            categorical_features = X.columns.tolist()
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            st.write(f"**Training set size:** {X_train.shape[0]} samples")
            st.write(f"**Test set size:** {X_test.shape[0]} samples")
            
            # Preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
                ]
            )
            
            # Model selection
            st.write("---")
            st.write(f"### 🤖 Training {model_choice} Model...")
            
            with st.spinner(f"Training {model_choice}..."):
                if model_choice == "Random Forest":
                    model = Pipeline([
                        ("preprocessor", preprocessor),
                        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
                    ])
                elif model_choice == "Logistic Regression":
                    model = Pipeline([
                        ("preprocessor", preprocessor),
                        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
                    ])
                else:  # Decision Tree
                    model = Pipeline([
                        ("preprocessor", preprocessor),
                        ("classifier", DecisionTreeClassifier(random_state=42))
                    ])
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                roc_auc = roc_auc_score(y_test, y_proba)
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                pr_auc = auc(recall, precision)
                
            st.success(f"✅ {model_choice} trained successfully!")
            
            # Display metrics
            st.write("### 📈 Model Performance")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("ROC-AUC Score", f"{roc_auc:.4f}")
            
            with metric_col2:
                st.metric("Precision-Recall AUC", f"{pr_auc:.4f}")
            
            with metric_col3:
                accuracy = (y_pred == y_test).mean()
                st.metric("Accuracy", f"{accuracy:.4f}")
            
            # Classification report and confusion matrix
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.write("#### Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.3f}"))
            
            with result_col2:
                st.write("#### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig3, ax3 = plt.subplots(figsize=(6,5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                           xticklabels=['Non-Veg', 'Vegetarian'],
                           yticklabels=['Non-Veg', 'Vegetarian'])
                ax3.set_xlabel('Predicted')
                ax3.set_ylabel('Actual')
                ax3.set_title('Confusion Matrix')
                st.pyplot(fig3)
            
            # Cross-validation
            with st.expander("🔄 Cross-Validation Results"):
                with st.spinner("Running 5-fold cross-validation..."):
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
                    
                    st.write(f"**CV ROC-AUC Mean:** {cv_scores.mean():.4f}")
                    st.write(f"**CV ROC-AUC Std:** {cv_scores.std():.4f}")
                    
                    fig4, ax4 = plt.subplots(figsize=(8,4))
                    ax4.plot(range(1, 6), cv_scores, marker='o', linestyle='-', color='#4ecdc4')
                    ax4.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
                    ax4.set_xlabel('Fold')
                    ax4.set_ylabel('ROC-AUC Score')
                    ax4.set_title('Cross-Validation Scores')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                    st.pyplot(fig4)
            
            # Feature importance (only for Random Forest and Decision Tree)
            if model_choice in ["Random Forest", "Decision Tree"]:
                with st.expander("🎯 Feature Importance (Top 20)"):
                    # Get feature names after one-hot encoding
                    feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
                    
                    # Get feature importances
                    importances = model.named_steps['classifier'].feature_importances_
                    
                    # Create dataframe and sort
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False).head(20)
                    
                    # Plot
                    fig5, ax5 = plt.subplots(figsize=(10,6))
                    ax5.barh(range(len(importance_df)), importance_df['importance'], color='#95e1d3')
                    ax5.set_yticks(range(len(importance_df)))
                    ax5.set_yticklabels(importance_df['feature'])
                    ax5.invert_yaxis()
                    ax5.set_xlabel('Importance')
                    ax5.set_title('Top 20 Feature Importances')
                    st.pyplot(fig5)

else:
    # Custom file upload
    st.info("👆 Please check 'Use default cheese dataset' in the sidebar to get started, or upload your own CSV file below.")
    uploaded_file = st.file_uploader("Upload your cheese dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(df.head())
        st.info("Custom dataset processing coming soon! For now, please use the default dataset.")

# Footer
st.write("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | Data source: <a href='https://github.com/rfordatascience/tidytuesday'>TidyTuesday</a></p>
</div>
""", unsafe_allow_html=True)