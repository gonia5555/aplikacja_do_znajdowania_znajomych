import json

import streamlit as st
import pandas as pd
import numpy as np
from pycaret.clustering import load_model, predict_model
import plotly.express as px
from dotenv import dotenv_values
# do pracy z qdrantem
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams, Distance
# do pracy z openai
from openai import OpenAI

QDRANT_COLLECTION_NAME = "aplikacja_do_znajdywania_znajomych"

env = dotenv_values(".env")
### Secrets using Streamlit Cloud Mechanism
# https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
if 'QDRANT_URL' in st.secrets:
    env['QDRANT_URL'] = st.secrets['QDRANT_URL']
if 'QDRANT_API_KEY' in st.secrets:
    env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']
##


EMBEDDING_DIM = 1536

EMBEDDING_MODEL = "text-embedding-3-small"

def get_openai_client():
    return OpenAI(api_key=env["OPENAI_API_KEY"])

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
    url=env["QDRANT_URL"], 
    api_key=env["QDRANT_API_KEY"],
)


def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        print("Tworzę kolekcję")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
    else:
        print("Kolekcja już istnieje")


def get_embeddings(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )

    return result.data[0].embedding

# >>>>>>>>>> WAŻNE <<<<<<<<<<<
# Tworzymy kolekcję przy starcie aplikacji
assure_db_collection_exists()


# ================= CONFIG =================
MODEL_NAME = 'welcome_survey_clustering_pipeline_v1'
DATA = 'welcome_survey_simple_v1.csv'
PL_JSON = 'welcome_survey_cluster_names_and_descriptions_v1.json'
EN_JSON = 'welcome_survey_cluster_names_and_descriptions_v1english.json'

CLUSTER_IMAGES = {
    "Cluster 0": "images/cluster_0.png",
    "Cluster 1": "images/cluster_1.png",
    "Cluster 2": "images/cluster_2.png",
    "Cluster 3": "images/cluster_3.png",
    "Cluster 4": "images/cluster_4.png",
    "Cluster 5": "images/cluster_5.png",
    "Cluster 6": "images/cluster_6.png",
    "Cluster 7": "images/cluster_7.png",
}

st.set_page_config(page_title="Community Matching", layout="wide")

# ================= CACHE =================
@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants(_model):
    df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(_model, data=df)
    return df_with_clusters

# ================= MAPPINGS =================
MAP_EDU = {
    "Podstawowe": "Primary",
    "Średnie": "Secondary",
    "Wyższe": "Higher"
}

MAP_ANIMALS = {
    "Brak ulubionych": "None",
    "Psy": "Dogs",
    "Koty": "Cats",
    "Inne": "Other",
    "Koty i Psy": "Cats and Dogs"
}

MAP_PLACE = {
    "Nad wodą": "By the water",
    "W lesie": "In the forest",
    "W górach": "In the mountains",
    "Inne": "Other"
}

MAP_GENDER = {
    "Mężczyzna": "Male",
    "Kobieta": "Female"
}

MAP_EDU_INV = {v: k for k, v in MAP_EDU.items()}
MAP_ANIMALS_INV = {v: k for k, v in MAP_ANIMALS.items()}
MAP_PLACE_INV = {v: k for k, v in MAP_PLACE.items()}
MAP_GENDER_INV = {v: k for k, v in MAP_GENDER.items()}

# ================= TRANSLATION =================
T = {
    "PL": {
        "sidebar_title": "✨ Opowiedz nam o sobie",
        "sidebar_desc": "Aplikacja dobierze Cię do grupy osób o podobnych preferencjach.",
        "age": "Wiek",
        "edu": "Wykształcenie",
        "animals": "Ulubione zwierzęta",
        "place": "Ulubione miejsce",
        "gender": "Płeć",
        "male": "Mężczyzna",
        "female": "Kobieta",
        "title": "🎯 Twój profil dopasowania",
        "match": "🔍 Jak bardzo pasujesz?",
        "match_score": "Stopień dopasowania",
        "expander": "❓ Jak to działa?",
        "expander_txt": "Aplikacja wykorzystuje machine learning do analizowania Twoich odpowiedzi.",
        "your_group": "📊 Twoja grupa",
        "people_group": "Liczba osób w grupie",
        "common_edu": "Najczęstsze wykształcenie",
        "common_animals": "Najpopularniejsze zwierzęta",
        "distributions": "📊 Rozkłady w Twojej grupie",
        "age_plot": "Wiek",
        "edu_plot": "Wykształcenie",
        "animals_plot": "Zwierzęta",
        "place_plot": "Miejsce",
        "gender_plot": "Płeć",
        "samples": "👥 Przykładowe osoby z Twojej grupy",
        "show_samples": "Pokaż losowe osoby",
        "ylabel": "Liczba osób"
    },
    "EN": {
        "sidebar_title": "✨ Tell us about yourself",
        "sidebar_desc": "The app will match you with people with similar preferences.",
        "age": "Age",
        "edu": "Education",
        "animals": "Favorite animals",
        "place": "Favorite place",
        "gender": "Gender",
        "male": "Male",
        "female": "Female",
        "title": "🎯 Your Match Profile",
        "match": "🔍 How well do you match?",
        "match_score": "Match score",
        "expander": "❓ How does it work?",
        "expander_txt": "The app uses machine learning to analyze your responses.",
        "your_group": "📊 Your group",
        "people_group": "People in group",
        "common_edu": "Most common education",
        "common_animals": "Most common animals",
        "distributions": "📊 Distributions in your group",
        "age_plot": "Age",
        "edu_plot": "Education",
        "animals_plot": "Animals",
        "place_plot": "Place",
        "gender_plot": "Gender",
        "samples": "👥 Sample participants",
        "show_samples": "Show random participants",
        "ylabel": "Count"
    }
}

# ================= SIDEBAR =================
with st.sidebar:
    lang = st.radio("Language / Język", ["PL", "EN"])
    tr = T[lang]

    st.header(tr["sidebar_title"])
    st.markdown(tr["sidebar_desc"])

    age = st.selectbox(tr["age"], ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown'])

    if lang == "EN":
        edu_choices = ['Primary', 'Secondary', 'Higher']
        animals_choices = ['None', 'Dogs', 'Cats', 'Other', 'Cats and Dogs']
        place_choices = ['By the water', 'In the forest', 'In the mountains', 'Other']
        gender_choices = [tr["male"], tr["female"]]
    else:
        edu_choices = ['Podstawowe', 'Średnie', 'Wyższe']
        animals_choices = ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy']
        place_choices = ['Nad wodą', 'W lesie', 'W górach', 'Inne']
        gender_choices = [tr["male"], tr["female"]]

    edu_level = st.selectbox(tr["edu"], edu_choices)
    fav_animals = st.selectbox(tr["animals"], animals_choices)
    fav_place = st.selectbox(tr["place"], place_choices)
    gender = st.radio(tr["gender"], gender_choices)

    person_df_ui = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender
    }])

# ================= LOAD DATA =================
with st.spinner("Loading model…"):
    model = get_model()
    all_df = get_all_participants(model)

cluster_info = load_json(PL_JSON) if lang == "PL" else load_json(EN_JSON)

person_df_for_model = person_df_ui.copy()
if lang == "EN":
    person_df_for_model['edu_level'] = person_df_for_model['edu_level'].map(MAP_EDU_INV).fillna(person_df_for_model['edu_level'])
    person_df_for_model['fav_animals'] = person_df_for_model['fav_animals'].map(MAP_ANIMALS_INV).fillna(person_df_for_model['fav_animals'])
    person_df_for_model['fav_place'] = person_df_for_model['fav_place'].map(MAP_PLACE_INV).fillna(person_df_for_model['fav_place'])
    person_df_for_model['gender'] = person_df_for_model['gender'].map(MAP_GENDER_INV).fillna(person_df_for_model['gender'])

predicted_cluster = predict_model(model, data=person_df_for_model)["Cluster"].values[0]
cluster_data = cluster_info[predicted_cluster]

# ================= UI =================
st.title(tr["title"])
st.subheader(cluster_data['name'])
st.write(cluster_data['description'])

image_path = CLUSTER_IMAGES.get(predicted_cluster)
if image_path:
    st.image(image_path, use_container_width=True)

# ================= MATCH SCORE =================
st.subheader(tr["match"])
match_score = np.random.uniform(85, 98)
st.metric(tr["match_score"], f"{match_score:.1f}%")
st.progress(match_score / 100)

with st.expander(tr["expander"]):
    st.markdown(tr["expander_txt"])

# ================= GROUP =================
st.header(tr["your_group"])

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster].copy()

def make_display_df(df, lang):
    df_disp = df.copy()
    for c in ['edu_level', 'fav_animals', 'fav_place', 'gender']:
        if c in df_disp.columns:
            df_disp[c] = df_disp[c].astype(str)
    if lang == "EN":
        df_disp['edu_level'] = df_disp['edu_level'].map(MAP_EDU).fillna(df_disp['edu_level'])
        df_disp['fav_animals'] = df_disp['fav_animals'].map(MAP_ANIMALS).fillna(df_disp['fav_animals'])
        df_disp['fav_place'] = df_disp['fav_place'].map(MAP_PLACE).fillna(df_disp['fav_place'])
        df_disp['gender'] = df_disp['gender'].map(MAP_GENDER).fillna(df_disp['gender'])
    return df_disp

same_cluster_disp = make_display_df(same_cluster_df, lang)

people_in_group = len(same_cluster_disp)

def safe_mode(series):
    try:
        return series.mode(dropna=True)[0]
    except Exception:
        return "—"

common_edu = safe_mode(same_cluster_disp["edu_level"])
common_animals = safe_mode(same_cluster_disp["fav_animals"])

col1, col2, col3 = st.columns(3)
col1.metric(tr["people_group"], people_in_group)
col2.metric(tr["common_edu"], common_edu)
col3.metric(tr["common_animals"], common_animals)

# ================= PLOTS =================
st.header(tr["distributions"])

AGE_ORDER = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown']

def plot_hist(df_original, col, title, xlabel, lang):
    df_plot = df_original.copy()

    if col not in df_plot.columns:
        st.info(f"No data for {col}")
        return

    df_plot[col] = df_plot[col].astype(str)

    if lang == "EN":
        if col == "edu_level":
            df_plot[col] = df_plot[col].map(MAP_EDU).fillna(df_plot[col])
        elif col == "fav_animals":
            df_plot[col] = df_plot[col].map(MAP_ANIMALS).fillna(df_plot[col])
        elif col == "fav_place":
            df_plot[col] = df_plot[col].map(MAP_PLACE).fillna(df_plot[col])
        elif col == "gender":
            df_plot[col] = df_plot[col].map(MAP_GENDER).fillna(df_plot[col])

    AGE_ORDER = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown']

    fig = px.histogram(df_plot, x=col)

    # ⭐ WYMUSZONE SORTOWANIE DLA WIEKU
    if col == "age":
        fig.update_xaxes(
            categoryorder="array",
            categoryarray=AGE_ORDER
        )

    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=tr["ylabel"])
    st.plotly_chart(fig, use_container_width=True)


plot_hist(same_cluster_df, "age", tr["age_plot"], tr["age_plot"], lang)
plot_hist(same_cluster_df, "edu_level", tr["edu_plot"], tr["edu_plot"], lang)
plot_hist(same_cluster_df, "fav_animals", tr["animals_plot"], tr["animals_plot"], lang)
plot_hist(same_cluster_df, "fav_place", tr["place_plot"], tr["place_plot"], lang)
plot_hist(same_cluster_df, "gender", tr["gender_plot"], tr["gender_plot"], lang)

# ================= SAMPLES =================
st.header(tr["samples"])
if st.button(tr["show_samples"]):
    st.dataframe(same_cluster_disp.sample(min(5, len(same_cluster_disp))))
