import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from imblearn.over_sampling import SMOTE
import streamlit as st

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv("IMDB-Movie-Data.csv")
    return data

# Preprocess Data
def preprocess_data(data):
    # Success Criteria
    median_revenue = data['Revenue (Millions)'].median(skipna=True)
    data['Success'] = (
        (data['Rating'] >= 7.0) &
        ((data['Revenue (Millions)'] >= median_revenue) | data['Revenue (Millions)'].isna())
    )

    # One-hot encoding
    mlb = MultiLabelBinarizer()
    actors_encoded = mlb.fit_transform(data['Actors'].str.split(','))
    actors_df = pd.DataFrame(actors_encoded, columns=mlb.classes_)
    directors_df = pd.get_dummies(data['Director'], prefix='Director')

    X = pd.concat([actors_df, directors_df], axis=1)
    y = data['Success']

    # Handle Imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled, mlb, directors_df.columns

# Train Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# App Interface
st.title("IMDB Movie Success Predictor")

# Load and preprocess data
data = load_data()
X, y, mlb, director_columns = preprocess_data(data)
model, _, _ = train_model(X, y)

# User Input
directors = data['Director'].unique()
actors = sorted(set(mlb.classes_))

director_input = st.selectbox("Choose a Director", directors)
actor1_input = st.selectbox("Choose Actor 1", actors)
actor2_input = st.selectbox("Choose Actor 2 (or None)", ["None"] + actors)
actor3_input = st.selectbox("Choose Actor 3 (or None)", ["None"] + actors)
actor4_input = st.selectbox("Choose Actor 4 (or None)", ["None"] + actors)

# Prepare Inputs for Prediction
selected_actors = [actor for actor in [actor1_input, actor2_input, actor3_input, actor4_input] if actor != "None"]
actor_vector = [1 if actor in selected_actors else 0 for actor in mlb.classes_]
director_vector = [1 if f"Director_{director_input}" == col else 0 for col in director_columns]
input_vector = actor_vector + director_vector

# Predict
if st.button("Predict"):
    prediction = model.predict([input_vector])
    result = "Success" if prediction[0] else "Flop"
    st.write(f"The predicted outcome for this movie is: **{result}**")
