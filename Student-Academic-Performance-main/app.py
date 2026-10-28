import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
data = pd.read_csv(r"C:\Users\vasan\OneDrive\Documents\Desktop\st data\Student_performance_data _.csv")

st.set_page_config(page_title="Student Scoring Dashboard", layout="wide")
st.title("📊 Student Performance Analysis & Scoring")

# EDA Section
st.header("📌 Exploratory Data Analysis (EDA)")

st.subheader("1. Dataset Overview")
st.dataframe(data.head())

st.subheader("2. Statistical Summary")
st.write(data.describe())

st.subheader("3. GPA Distribution")
fig, ax = plt.subplots()
sns.histplot(data["GPA"], bins=20, kde=True, ax=ax)
st.pyplot(fig)

st.subheader("4. Absences vs GPA")
fig2, ax2 = plt.subplots()
sns.scatterplot(x="Absences", y="GPA", data=data, ax=ax2)
st.pyplot(fig2)

st.subheader("5. Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

# Show column names for debugging
st.write('Columns in data:', data.columns.tolist())

# Scoring Section
st.header("📈 Student Scoring Based on Performance Criteria")

def calculate_score(row):
    score = 0

    # StudyTime
    if row['StudyTimeWeekly'] > 15:
        score += 4
    elif row['StudyTimeWeekly'] > 10:
        score += 3
    elif row['StudyTimeWeekly'] > 5:
        score += 2
    else:
        score += 1

    # Absences
    if row['Absences'] < 5:
        score += 3
    elif row['Absences'] < 15:
        score += 2
    else:
        score += 1

    # Extracurricular
    extra = row['Sports'] + row['Music']
    if extra == 2:
        score += 3
    elif extra == 1:
        score += 2
    else:
        score += 1

    # Tutoring and Volunteering
    score += row['Tutoring']
    score += row['Volunteering']

    return score

data['score'] = data.apply(calculate_score, axis=1)

def label_performance(score):
    if score >= 11:
        return "Excellent"
    elif score >= 9:
        return "Good"
    elif score >= 7:
        return "Average"
    else:
        return "Low"

data['performance'] = data['score'].apply(label_performance)

# Display results by group
for category in ['Excellent', 'Good', 'Average', 'Low']:
    st.subheader(f"🔹 {category} Performers")
    st.dataframe(
        data[data['performance'] == category][
            ['StudyTimeWeekly', 'Absences', 'Sports', 'Music', 'Tutoring', 'Volunteering', 'score', 'performance']
        ]
    )

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

st.header("🤖 Model Training and Evaluation")

# Label encode the performance column
le = LabelEncoder()
data['performance_encoded'] = le.fit_transform(data['performance'])  # Target: 0=Average, 1=Excellent, etc.

# Features and target
features = ['StudyTimeWeekly', 'Absences', 'Sports', 'Music', 'Tutoring', 'Volunteering']
target = 'performance_encoded'

# Custom split: first 80% train, last 20% test
split_idx = int(len(data) * 0.8)
train_data = data.iloc[:split_idx]
test_data = data.iloc[split_idx:]

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

# Train an XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred) * 100
test_acc = accuracy_score(y_test, y_test_pred) * 100

st.success(f"✅ Training Accuracy: {train_acc:.2f}%")
st.success(f"✅ Testing Accuracy: {test_acc:.2f}%")

# Show predictions in test set
st.subheader("📋 Predicted vs Actual (Test Data)")
test_results = test_data.copy()
test_results['Predicted'] = le.inverse_transform(y_test_pred)
test_results['Actual'] = le.inverse_transform(y_test)
st.dataframe(test_results[['StudyTimeWeekly', 'Absences', 'Sports', 'Music', 'Tutoring', 'Volunteering', 'Actual', 'Predicted']])



import plotly.express as px

# Count of each performance category
performance_counts = data['performance'].value_counts().reset_index()
performance_counts.columns = ['Performance Category', 'Count']

st.subheader("📊 Student Performance Category Distribution")

# Plotly bar chart
fig = px.bar(
    performance_counts,
    x='Performance Category',
    y='Count',
    color='Performance Category',
    title='Number of Students in Each Performance Category',
    text='Count',
    template='plotly_dark'
)
fig.update_traces(textposition='outside')
fig.update_layout(showlegend=False)

st.plotly_chart(fig, use_container_width=True)

st.header("🔍 Data Quality Check")

st.subheader("1. Missing Values")
missing_values = data.isnull().sum()
st.write(missing_values[missing_values > 0] if missing_values.sum() > 0 else "✅ No missing values found.")


st.subheader("2. Data Types")
st.write(data.dtypes)

st.sidebar.header("🎓 Predict a New Student's Performance")
study = st.sidebar.slider("Study Time Weekly", 0, 25, 10)
absences = st.sidebar.slider("Absences", 0, 50, 5)
sports = st.sidebar.radio("Sports", [0, 1])
music = st.sidebar.radio("Music", [0, 1])
tutoring = st.sidebar.radio("Tutoring", [0, 1])
volunteering = st.sidebar.radio("Volunteering", [0, 1])

# Prepare input
new_data = pd.DataFrame([[study, absences, sports, music, tutoring, volunteering]], columns=features)
pred = model.predict(new_data)
pred_label = le.inverse_transform(pred)[0]
st.sidebar.success(f"📈 Predicted Performance: {pred_label}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plotly.express as px

st.header("⚖️ Model Comparison")

models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

accuracy_scores = {}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    accuracy_scores[name] = acc
    st.write(f"🔹 {name} Accuracy: **{acc:.2f}%**")

# Plotting
acc_df = pd.DataFrame(list(accuracy_scores.items()), columns=['Model', 'Accuracy'])
fig = px.bar(acc_df, x='Model', y='Accuracy', color='Model', text='Accuracy',
             title="📊 Model Accuracy Comparison", template='plotly_dark')
fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)


export_cols = ['StudyTimeWeekly', 'Absences', 'Sports', 'Music', 'Tutoring', 'Volunteering',
               'score', 'performance', 'performance_encoded']

data[export_cols].to_csv("final_dashboard_data.csv", index=False)
# Add a download button in Streamlit
st.download_button(
    label="Download Processed Data as CSV",
    data=data[export_cols].to_csv(index=False).encode('utf-8'),
    file_name='final_dashboard_data.csv',
    mime='text/csv'
)
