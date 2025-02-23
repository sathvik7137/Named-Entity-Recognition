import streamlit as st
import spacy
from spacy import displacy
import pandas as pd
import json
import os

nlp = spacy.load('en_core_web_sm')

# Function to perform Named Entity Recognition
def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
    return entities

# Function to visualize named entities
def visualize_entities(text):
    doc = nlp(text)
    displacy.render(doc, style="ent", jupyter=False, page=True)

# Function to convert entities to a pandas DataFrame
def entities_to_dataframe(entities):
    return pd.DataFrame(entities, columns=["Entity", "Label", "Start Char", "End Char"])

# Function to save extracted entities as a CSV or JSON file
def save_entities(entities, file_type='csv'):
    df = entities_to_dataframe(entities)
    if file_type == 'csv':
        df.to_csv('entities.csv', index=False)
        return 'entities.csv'
    elif file_type == 'json':
        df.to_json('entities.json', orient='records', lines=True)
        return 'entities.json'
    else:
        return None

# Streamlit App UI
st.title("Named Entity Recognition with SpaCy")
st.write("""
This app allows you to extract named entities from text using SpaCy's pre-trained models. 
You can either type or upload a text, and the app will identify the entities (e.g., persons, organizations, locations, etc.).
""")

# Text input or file upload
text_option = st.radio("Choose input type", ["Enter Text", "Upload File"])

if text_option == "Enter Text":
    user_input = st.text_area("Input Text", "Enter your text here...")
else:
    uploaded_file = st.file_uploader("Choose a text file", type="txt")
    if uploaded_file is not None:
        user_input = uploaded_file.read().decode("utf-8")
        st.text_area("Uploaded Text", user_input, height=200)

# Option to choose model for NER
model_choice = st.selectbox("Choose NER Model", ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg'])
if model_choice != 'en_core_web_sm':
    nlp = spacy.load(model_choice)

# Button to analyze text
if st.button("Analyze"):
    if user_input:
        # Perform NER
        entities = perform_ner(user_input)
        
        # Show extracted entities in the app
        if entities:
            st.write("### Extracted Named Entities:")
            entity_df = entities_to_dataframe(entities)
            st.dataframe(entity_df)
        
            # Statistics of entities
            st.write("### Entity Statistics:")
            st.write(entity_df['Label'].value_counts())
        
            # Save results button
            save_option = st.radio("Save Results", ["No", "CSV", "JSON"])
            if save_option != "No":
                file_name = save_entities(entities, save_option.lower())
                st.success(f"Results saved as {file_name}")
        
            # Visualize the named entities
            st.write("### Named Entity Visualization:")
            visualize_entities(user_input)
        else:
            st.write("No named entities found.")
    else:
        st.write("Please enter or upload some text to analyze.")
