# Análisis de Reseñas de Mercado Libre 📦🔍

## 0) Dependencias 📚
import re
import io

import pandas as pd
import numpy as np

from transformers import pipeline
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import gradio as gr


## 1) NLP Pipelines 🙌

# Pipeline de nuestro finetuned model para análisis de sentimiento
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cabustillo13/roberta-base-bne-platzi-project-nlp-con-transformers"
)

# Pipeline para NER en español
ner_pipeline = pipeline(
    "ner",
    model="mrm8488/bert-spanish-cased-finetuned-ner",
    tokenizer="mrm8488/bert-spanish-cased-finetuned-ner"
)

## 2) Funcionalidad 🦾

# Función para limpiar el texto
def clean(text):
    # Eliminar textos entre corchetes (ej.: etiquetas)
    text = re.sub(r'\[.*?\]', '', text)

    # Eliminar URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Eliminar etiquetas HTML
    text = re.sub(r'<.*?>+', '', text)

    # Eliminar espacios extras al inicio y final
    text = text.strip()

    return text

# Función para reconstruir una entidad a partir de los tokens del NER
def reconstruct_entity(ner_tokens):
    """
    Reconstruye una entidad a partir de una lista de tokens de NER.
    Si un token empieza con "##", se une al token anterior sin espacio.
    """
    entity = ""
    for token in ner_tokens:
        word = token['word']
        if word.startswith("##"):
            entity += word[2:]
        else:
            if entity:
                entity += " " + word
            else:
                entity += word
    return entity

# Función para procesar la salida del NER y agrupar tokens en entidades completas
def process_ner_output(ner_results):
    """
    Procesa la salida del NER ignorando el tipo de entidad y devuelve un diccionario
    con una única clave "entities" cuyo valor es la entidad reconstruida a partir de todos los tokens.
    """
    # Reconstruir la entidad a partir de todos los tokens de la lista
    combined = reconstruct_entity(ner_results)
    return {"entities": combined}

# Función para analizar un solo texto
def analyze_text(input_text):
    input_text = clean(input_text)
    sentiment = sentiment_pipeline(input_text)
    ner_results = ner_pipeline(input_text)
    processed_ner = process_ner_output(ner_results)
    return sentiment, processed_ner

# Función para analizar un archivo CSV
def analyze_csv(file_obj):
    df = pd.read_csv(file_obj.name)
    if "review_body" not in df.columns:
        return "Error: No se encontró la columna 'review_body'.", None, None
    texts = df["review_body"].astype(str).tolist()

    # Limpiar cada reseña
    cleaned_texts = [clean(text) for text in texts]

    # Obtener análisis de sentimiento y NER para cada reseña limpia
    sentiments = [sentiment_pipeline(text) for text in cleaned_texts]
    ner_all = [process_ner_output(ner_pipeline(text)) for text in cleaned_texts]

    # Extraer las entidades detectadas (valor) de cada reseña
    ner_words = []
    for ner_result in ner_all:
        # ner_result es un diccionario con la clave "entities"
        ner_words.append(ner_result["entities"])

    # Unir todas las entidades en un solo string
    combined_ner_text = " ".join(ner_words)

    # Generar wordcloud basado en las entidades detectadas
    wc = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=400).generate(combined_ner_text)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)

    # Convertir a imagen PIL para que Gradio lo pueda mostrar
    image = Image.open(buf)

    return sentiments, ner_all, image

## 3) Interfaz Gráfica ✨

"""**Construir la interfaz**"""

with gr.Blocks(theme=gr.themes.Citrus()) as demo:
    gr.Markdown("## Aplicación de Análisis de Reseñas de Mercado Libre 📦🔍")

    with gr.Tab("Análisis de Texto"):
        gr.Markdown("### Ingrese una reseña de texto")
        text_input = gr.Textbox(label="Texto de Reseña", placeholder="Escribe aquí la reseña...")
        sentiment_output = gr.JSON(label="Análisis de Sentimiento")
        ner_output = gr.JSON(label="Entidades Reconocidas (NER)")
        analyze_btn = gr.Button("Analizar Texto")
        analyze_btn.click(analyze_text, inputs=text_input, outputs=[sentiment_output, ner_output])

    with gr.Tab("Análisis de CSV"):
        gr.Markdown("### Suba un archivo CSV con una columna 'review_body'")
        csv_input = gr.File(label="Archivo CSV")
        csv_sentiment_output = gr.JSON(label="Análisis de Sentimiento (por Reseña)")
        csv_ner_output = gr.JSON(label="Entidades Reconocidas (por Reseña)")
        wc_output = gr.Image(label="WordCloud (Entidades)")
        analyze_csv_btn = gr.Button("Analizar CSV")
        analyze_csv_btn.click(analyze_csv, inputs=csv_input, outputs=[csv_sentiment_output, csv_ner_output, wc_output])

## 4) Ready! 🚀

#**Iniciar instancia**
demo.launch()
