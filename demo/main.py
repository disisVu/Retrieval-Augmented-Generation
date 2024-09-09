import dash
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from langdetect import detect

import os
from transformers import (
  MarianTokenizer,
  MarianMTModel,
  DPRQuestionEncoderTokenizer,
  DPRQuestionEncoder,
  T5Tokenizer, 
  T5ForConditionalGeneration,
  AutoTokenizer, 
  AutoModelForSequenceClassification
)
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


external_document_dir = os.path.join('external')

embedding_tokenizer, embedding_model, generation_tokenizer, generation_model, ranking_tokenizer, ranking_model, translation_tokenizer, translation_model = [None] * 8


# Initialize the Flask server for Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


def initialize_models():
  global embedding_tokenizer, embedding_model, generation_tokenizer, generation_model, ranking_tokenizer, ranking_model, translation_tokenizer, translation_model

  # Initialize tokenizer and model for embedding generation
  embedding_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
  embedding_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

  # Initialize tokenizer and model for text generation
  generation_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
  generation_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

  # Initialize tokenizer and model for re-ranking
  ranking_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
  ranking_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

  # Initialize tokenizer and model for translation Vietnamese-English
  translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
  translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-vi-en")

  return embedding_tokenizer, embedding_model, generation_tokenizer, generation_model, ranking_tokenizer, ranking_model, translation_tokenizer, translation_model


def translate_to_english(query):
  global translation_tokenizer, translation_model
  inputs = translation_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
  translated_tokens = translation_model.generate(**inputs)
  translated_texts = [translation_tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
  return translated_texts


def detect_and_translate_text(texts):
  translated_texts = []
  for text in texts:
    if detect(text) != 'en':
      translated_texts.append(translate_to_english(text))
    else:
      translated_texts.append(text)
  return translated_texts


def chunk_text(text, tokenizer, max_length=512):
  # Tokenize the entire text without truncation
  tokens = tokenizer.encode(text, add_special_tokens=False)
  
  # Split tokens into chunks of max_length
  chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
  
  # Decode tokens back to text for each chunk
  chunk_texts = [tokenizer.decode(chunk, clean_up_tokenization_spaces=True) for chunk in chunks]
  
  # # Detect and translate non-English chunks
  # translated_chunks = detect_and_translate_text(chunk_texts)
  return chunk_texts


# Read Document .txt Files
def get_data_from_txt_files(folder_path, tokenizer, max_length=512):
  documents = []  # List to store the content of each file
  
  # Loop through all files in the folder
  for filename in os.listdir(folder_path):
    # Check if the file is a .txt file
    if filename.endswith('.txt'):
      file_path = os.path.join(folder_path, filename)
      
      # Open and read the content of the file
      with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # Chunk the content
        chunks = chunk_text(content, tokenizer, max_length)
        documents.extend(chunks)
  
  return documents


def generate_embeddings(texts, tokenizer, model, max_length=512):
    all_embeddings = []

    for text in texts:
        # Tokenize text with padding and truncation to ensure uniform input size
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,               # Truncate sequences longer than max_length
            padding='max_length',          # Pad sequences to max_length
            max_length=max_length          # Set maximum length to 512 tokens
        )

        # Generate embeddings using the DPR model
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.pooler_output  # Use the pooler_output for DPR models
        
        all_embeddings.append(embedding)

    # Ensure the embeddings are a 2D tensor: [num_texts, embedding_dim]
    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0)



# Rank Documents Using Embeddings of Query and External Documents
def rank_documents(query_embedding, doc_embeddings):
  # Compute cosine similarity between query and document embeddings
  cos_similarities = cosine_similarity(query_embedding, doc_embeddings)

  # Get the indices of documents sorted by similarity (descending order)
  ranked_indices = np.argsort(-cos_similarities, axis=1)

  return ranked_indices, cos_similarities


# Retrieve top-K documents
def retrieve_top_k_documents(query_embedding, doc_embeddings, documents, k=5):
  ranked_indices, cos_similarities = rank_documents(query_embedding, doc_embeddings)
  top_k_indices = ranked_indices[0][:k]
  top_k_documents = [documents[idx] for idx in top_k_indices]
  return top_k_documents


# Re-Rank the Top-K Documents
def re_rank_documents(query, top_k_documents, ranking_tokenizer, ranking_model):
  inputs = [f"{query} [SEP] {doc}" for doc in top_k_documents]
  tokenized_inputs = ranking_tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
  with torch.no_grad():
    outputs = ranking_model(**tokenized_inputs)
  scores = outputs.logits.squeeze().tolist()
  ranked_docs = [doc for _, doc in sorted(zip(scores, top_k_documents), reverse=True)]
  print("\nRanked Documents:")
  for i, doc in enumerate(ranked_docs):
    print(f"Rank {i+1}: Score={scores[i]}, Document: {doc[:100]}...")
  return ranked_docs


# Generate Response
def generate_response(query, top_k_documents, generation_tokenizer, generation_model):
  # Concatenate top-K documents into a single context
  context = " ".join(top_k_documents)
  input_text = query + " " + context

  inputs = generation_tokenizer.encode(input_text, return_tensors="pt", truncation=True)

  # Debug: Print input length and contents
  print("\n=== Debug: Generate Response ===")
  print(f"Combined input length: {len(inputs[0])}")
  print(f"Input (truncated): {generation_tokenizer.decode(inputs[0], skip_special_tokens=True)}")

  with torch.no_grad():
    output = generation_model.generate(inputs, max_length=100, num_return_sequences=1)

  response = generation_tokenizer.decode(output[0], skip_special_tokens=True)

  # Debug: Print generated response
  print(f"Generated Response: {response}")
  return response


# Initialize chatbot
def initialize_chatbot(external_document_dir, max_length=512):
    # Initialize models and tokenizers
    embedding_tokenizer, embedding_model, generation_tokenizer, generation_model, ranking_tokenizer, ranking_model, translation_tokenizer, translation_model = initialize_models()
    
    # Get documents from text files and chunk them
    documents = get_data_from_txt_files(external_document_dir, embedding_tokenizer, max_length)

    # Generate embeddings for the documents
    doc_embeddings = generate_embeddings(documents, embedding_tokenizer, embedding_model)

    return embedding_tokenizer, embedding_model, generation_tokenizer, generation_model, ranking_tokenizer, ranking_model, translation_tokenizer, translation_model, doc_embeddings, documents


# Initialize the chatbot components
embedding_tokenizer, embedding_model, generation_tokenizer, generation_model, ranking_tokenizer, ranking_model, translation_tokenizer, translation_model, doc_embeddings, documents = initialize_chatbot(external_document_dir)

# Print each document
for document in documents:
    # Check if document is a list and join it into a single string if necessary
    if isinstance(document, list):
        document_text = ' '.join(document)
    else:
        document_text = document

    # Print the document
    print(document_text + '\n')

# Dash app layout
app.layout = dbc.Container(
  [
    dbc.Row(
      dbc.Col(
        html.H1("RAG Chatbot Application Interface", className="text-center my-4")
      )
    ),
    dbc.Row(
      dbc.Col(
        html.Div(id="chat-window")
      )
    ),
    dbc.Row(
      dbc.Col(
        dbc.Input(id="user-input", placeholder="Input your query here...", type="text", style={"width": "100%"})
      )
    ),
    dbc.Row(
      dbc.Col(
        dbc.Button("Send", id="send-button", color="primary", className="mt-3", n_clicks=0)
      )
    )
  ],
  fluid=True
)

# Define callback to update chatbot response
@app.callback(
  Output("chat-window", "children"),
  Input("send-button", "n_clicks"),
  State("user-input", "value"),
  State("chat-window", "children")
)
def update_output(n_clicks, user_message, chat_history):
  if n_clicks > 0 and user_message:
    # Add new message to the existing chat history
    if chat_history is None:
      chat_history = []

    # Add query message box
    new_query_box = html.Div(
      html.Div(
        html.Span(f"{user_message}"),
        className="message-box query"
      ),
      className="message-wrapper"
    )
    chat_history.append(new_query_box)

    # Generate chatbot response
    # translated_query = detect_and_translate_text(user_message)
    query_embedding = generate_embeddings([user_message], embedding_tokenizer, embedding_model)
    top_k_documents = retrieve_top_k_documents(query_embedding, doc_embeddings, documents, k=3)
    ranked_docs = re_rank_documents(user_message, top_k_documents, ranking_tokenizer, ranking_model)
    response = generate_response(user_message, ranked_docs, generation_tokenizer, generation_model)

    # Add response message box
    new_response_box = html.Div(
      html.Div(
        html.Span(f"{response}"),
        className="message-box response"
      ),
      className="message-wrapper"
    )
    chat_history.append(new_response_box)

    return chat_history

  return []

# Run the Dash app
if __name__ == "__main__":
  app.run_server(debug=True)