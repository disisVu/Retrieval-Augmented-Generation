import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import os
from transformers import AutoTokenizer, AutoModel, BartForConditionalGeneration
import torch
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


external_document_dir = os.path.join('external')  


# Initialize the Flask server for Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


#  Initialize Models and Tokenizers
def initialize_models():
  # Initialize tokenizer and model for embedding generation
  embedding_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
  embedding_model = AutoModel.from_pretrained("distilbert-base-uncased")

  # Initialize tokenizer and model for text generation
  generation_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
  generation_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

  return embedding_tokenizer, embedding_model, generation_tokenizer, generation_model


# Generate Embeddings
def generate_embeddings(texts, tokenizer, model):
  inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
  with torch.no_grad():
    outputs = model(**inputs)

  # Use the mean of the token embeddings as the sentence embedding
  embeddings = outputs.last_hidden_state.mean(dim=1)

  return embeddings


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


# Generate Response
def generate_response(query, top_k_documents, generation_tokenizer, generation_model):
  # Concatenate top-K documents into a single context
  context = " ".join(top_k_documents)
  input_text = query + " " + context

  inputs = generation_tokenizer.encode(input_text, return_tensors="pt", truncation=True)
  with torch.no_grad():
    output = generation_model.generate(inputs, max_length=100, num_return_sequences=1)

  response = generation_tokenizer.decode(output[0], skip_special_tokens=True)
  return response


# Read Document .txt Files
def get_data_from_txt_files(folder_path):
  documents = []  # List to store the content of each file
  
  # Loop through all files in the folder
  for filename in os.listdir(folder_path):
    # Check if the file is a .txt file
    if filename.endswith('.txt'):
      file_path = os.path.join(folder_path, filename)
      
      # Open and read the content of the file
      with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        documents.append(content)
  
  return documents


# Initialize chatbot
def initialize_chatbot():
  embedding_tokenizer, embedding_model, generation_tokenizer, generation_model = initialize_models()
  documents = get_data_from_txt_files(external_document_dir)

  # Generate embeddings for documents
  doc_embeddings = generate_embeddings(documents, embedding_tokenizer, embedding_model)
  
  return embedding_tokenizer, embedding_model, generation_tokenizer, generation_model, doc_embeddings, documents


# Initialize the chatbot components
embedding_tokenizer, embedding_model, generation_tokenizer, generation_model, doc_embeddings, documents = initialize_chatbot()


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
        html.Div(id="chat-history")
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
  Output("chat-history", "children"),
  Input("send-button", "n_clicks"),
  State("user-input", "value"),
  State("chat-history", "children")
)
def update_output(n_clicks, user_message, chat_history):
  if n_clicks > 0 and user_message:
    # Add new message to the existing chat history
    if chat_history is None:
      chat_history = []

    # Add query message box
    new_query_box = html.Div(
      [
        html.Div(
          html.Span(f"{user_message}"),
          className="message-box user-query-box"
        )
      ],
      className="message-row"
    )
    chat_history.append(new_query_box)

    # Generate chatbot response
    query_embedding = generate_embeddings([user_message], embedding_tokenizer, embedding_model)
    top_k_documents = retrieve_top_k_documents(query_embedding, doc_embeddings, documents, k=5)
    response = generate_response(user_message, top_k_documents, generation_tokenizer, generation_model)

    # Update chat history
    new_response_box = html.Div(
      [
        html.Div(
          html.Span(f"{response}"),
          className="message-box chatbot-response-box"
        )
      ],
      className="message-row"
    )
    chat_history.append(new_response_box)

    return chat_history

  return []

# Run the Dash app
if __name__ == "__main__":
  app.run_server(debug=True)