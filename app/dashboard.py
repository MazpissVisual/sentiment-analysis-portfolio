"""
Dashboard Streamlit — Sentiment Analysis Bahasa Indonesia
Jalankan dengan: streamlit run app/dashboard.py
"""

import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

RANDOM_STATE = 42


@st.cache_resource
def get_preprocessor():
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    stemmer = StemmerFactory().create_stemmer()
    return stopword_remover, stemmer


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df


@st.cache_resource
def train_model(df):
    train_df = df[df["split"] == "train"]
    valid_df = df[df["split"] == "valid"]

    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=3)
    X_train = tfidf.fit_transform(train_df["text_clean"])
    y_train = train_df["label"]

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return model, tfidf


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text, stopword_remover, stemmer):
    text = clean_text(text)
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text


st.title("Sentiment Analysis Dashboard — Bahasa Indonesia")
st.caption("Rumah Digicraft — Data Science Portfolio Project | Dataset: IndoNLU SmSA")

data_path = st.sidebar.text_input(
    "Path dataset (CSV)", value="data/indonlu_smsa_sentiment.csv"
)

try:
    df = load_data(data_path)
except FileNotFoundError:
    st.error(f"File tidak ditemukan: {data_path}. Sesuaikan path di sidebar.")
    st.stop()

model, tfidf = train_model(df)
stopword_remover, stemmer = get_preprocessor()

tab1, tab2, tab3 = st.tabs(["Overview Dataset", "Kata Kunci per Sentimen", "Coba Prediksi"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data", f"{len(df):,}")
    col2.metric("Rata-rata Panjang Teks", f"{df['text'].apply(lambda x: len(str(x).split())).mean():.1f} kata")
    dominant = df["label"].value_counts(normalize=True).idxmax()
    col3.metric("Sentimen Dominan", dominant.capitalize())

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        order = ["positive", "neutral", "negative"]
        colors = ["#55A868", "#8172B2", "#C44E52"]
        counts = df["label"].value_counts().reindex(order)
        ax.bar(order, counts.values, color=colors)
        ax.set_title("Distribusi Label Sentimen")
        ax.set_ylabel("Jumlah Data")
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots()
        df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
        sns.boxplot(data=df, x="label", y="word_count", order=order, palette=colors, ax=ax)
        ax.set_title("Panjang Teks per Sentimen")
        st.pyplot(fig)

    st.subheader("Contoh Data")
    sample_label = st.selectbox("Filter berdasarkan sentimen", ["Semua"] + order)
    if sample_label == "Semua":
        st.dataframe(df[["text", "label"]].sample(10, random_state=1), use_container_width=True)
    else:
        st.dataframe(
            df[df["label"] == sample_label][["text", "label"]].sample(
                min(10, len(df[df["label"] == sample_label])), random_state=1
            ),
            use_container_width=True,
        )

with tab2:
    st.subheader("Kata Paling Berpengaruh per Sentimen (dari model)")
    import numpy as np

    feature_names = np.array(tfidf.get_feature_names_out())
    classes = model.classes_

    cols = st.columns(len(classes))
    for col, cls in zip(cols, classes):
        idx = list(classes).index(cls)
        coefs = model.coef_[idx]
        top_idx = np.argsort(coefs)[-12:][::-1]
        with col:
            st.markdown(f"**{cls.capitalize()}**")
            for word in feature_names[top_idx]:
                st.write(f"- {word}")

with tab3:
    st.subheader("Coba Prediksi Sentimen Teks Baru")
    st.caption("Masukkan komentar atau review dalam Bahasa Indonesia untuk melihat prediksi sentimennya.")

    user_text = st.text_area(
        "Teks komentar/review",
        value="pelayanannya ramah banget dan makanannya enak, aku suka!",
        height=100,
    )

    if st.button("Prediksi Sentimen"):
        cleaned = preprocess(user_text, stopword_remover, stemmer)
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        proba_df = pd.DataFrame({"Sentimen": model.classes_, "Probabilitas": proba}).sort_values(
            "Probabilitas", ascending=False
        )

        color_map = {"positive": "🟢", "neutral": "🟣", "negative": "🔴"}
        st.markdown(f"### Prediksi: {color_map.get(pred, '')} **{pred.upper()}**")

        fig, ax = plt.subplots(figsize=(6, 3))
        colors_map = {"positive": "#55A868", "neutral": "#8172B2", "negative": "#C44E52"}
        bar_colors = [colors_map.get(s, "#333") for s in proba_df["Sentimen"]]
        ax.barh(proba_df["Sentimen"], proba_df["Probabilitas"], color=bar_colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilitas")
        st.pyplot(fig)

        with st.expander("Lihat hasil preprocessing teks"):
            st.write("**Teks asli:**", user_text)
            st.write("**Setelah cleaning + stopword removal + stemming:**", cleaned)

    st.markdown("---")
    st.subheader("Batch Prediksi (beberapa komentar sekaligus)")
    batch_text = st.text_area(
        "Masukkan beberapa komentar (1 baris = 1 komentar)",
        value="produknya bagus banget, aku suka!\npelayanan lambat dan mengecewakan\nbiasa aja sih menurutku",
        height=100,
    )
    if st.button("Prediksi Batch"):
        lines = [l.strip() for l in batch_text.split("\n") if l.strip()]
        cleaned_lines = [preprocess(l, stopword_remover, stemmer) for l in lines]
        vecs = tfidf.transform(cleaned_lines)
        preds = model.predict(vecs)

        result_df = pd.DataFrame({"Teks": lines, "Prediksi Sentimen": preds})
        st.dataframe(result_df, use_container_width=True)

        summary = result_df["Prediksi Sentimen"].value_counts(normalize=True) * 100
        st.write("**Ringkasan proporsi sentimen:**")
        st.write(summary.round(1).astype(str) + "%")

st.sidebar.markdown("---")
st.sidebar.info(
    "Dashboard ini bagian dari portfolio Data Science — "
    "klasifikasi sentimen teks Bahasa Indonesia menggunakan TF-IDF + Logistic Regression."
)
