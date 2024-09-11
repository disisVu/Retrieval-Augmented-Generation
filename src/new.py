import dash
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
from tqdm import tqdm
import tensorflow as tf
from transformers import (
  TFAutoModel,
  AutoTokenizer,
  T5Tokenizer,
  AutoModelForSequenceClassification,
  MT5ForConditionalGeneration,
)
import py_vncorenlp
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Thiết lập đường dẫn cache trên ổ D
os.environ['HF_HOME'] = 'D:/HuggingFace/cache'

# Initialize the Flask server for Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Khai báo các biến toàn cục
rdrsegmenter = None
embedding_model = None
embedding_tokenizer = None
generation_model = None
generation_tokenizer = None
ranking_model = None
ranking_tokenizer = None
documents = None
segmented_documents = None
doc_embeddings = None
k_count = 5
max_length = 256


current_dir = os.path.dirname(os.path.abspath(__file__))
#external_document_dir = os.path.join(current_dir, 'newspaper_data')
external_document_dir = os.path.join(current_dir, 'small_data')
vncorenlp_dir = os.path.join(current_dir, 'vncorenlp')
embeddings_file_path = os.path.join(current_dir, 'vectordb') + "\small_doc_embeddings.npy"

# BƯỚC 0:
# Khởi tạo các mô hình cần thiết
def initialize_models():
  global rdrsegmenter
  rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_dir)

  global embedding_model, embedding_tokenizer
  embedding_model = TFAutoModel.from_pretrained("vinai/phobert-large")
  embedding_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")

  global generation_model, generation_tokenizer
  generation_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large")
  generation_tokenizer = T5Tokenizer.from_pretrained("google/mt5-large")

  global ranking_model, ranking_tokenizer
  ranking_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
  ranking_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

# BƯỚC 1:
# Đọc dữ liệu và chia nhỏ văn bản từ file
# Read Document .txt Files
def read_file(file_path):
  with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()
    return chunk_text(content)

def get_data_from_txt_files(folder_path):
  documents = []
  # Dùng os.walk để lấy danh sách file
  file_paths = []
  for root, dirs, files in os.walk(folder_path):
    for filename in files:
      if filename.endswith('.txt'):
        file_paths.append(os.path.join(root, filename))

  # Xử lý tuần tự các file
  for file_path in file_paths:
    chunks = read_file(file_path)
    documents.extend(chunks)

  return documents

# BƯỚC 1.1:
# Chia nhỏ văn bản
def chunk_text(text):
  # Mã hóa văn bản
  tokens = embedding_tokenizer.encode(text, add_special_tokens=False)
  # Chia các đoạn mã hóa thành nhiều cụm nhỏ với độ dài cho trước
  chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
  # Giải mã về dạng dữ liệu văn bản
  chunk_texts = [embedding_tokenizer.decode(chunk, clean_up_tokenization_spaces=True) for chunk in chunks]
  return chunk_texts

# BƯỚC 2:
# Phân tách câu
# Ví dụ:
# input: "Chúng tôi là nghiên cứu viên"
# output: "Chúng tôi là nghiên_cứu viên"
def segment_words(text):
  return rdrsegmenter.word_segment(text)

def reverse_segment_words(segmented_text):
  text = ' '.join(segmented_text)
  return text.replace('_', ' ')

# BƯỚC 3:
# Tạo vector embedding từ văn bản đã được phân tách
# input: dữ liệu văn bản
# output: danh sách vector embedding
# Hàm generate embedding cho mỗi văn bản
def generate_single_embedding(text):
  inputs = embedding_tokenizer(text, return_tensors="tf", truncation=True, padding='max_length', max_length=max_length)
  outputs = embedding_model(**inputs)
  embedding = outputs.pooler_output
  return embedding[0]

# Hàm generate embeddings với threading và thanh tiến trình
def generate_embeddings(texts, num_threads=1):
  all_embeddings = []
  with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(generate_single_embedding, text) for text in texts]
    # Sử dụng tqdm để hiển thị thanh tiến trình cho các tác vụ song song
    for future in tqdm(as_completed(futures), total=len(texts), desc="Generating Embeddings"):
      embedding = future.result()
      all_embeddings.append(embedding)
  # Trả về các embedding dưới dạng TensorFlow tensor
  concatenated_embeddings = tf.concat(all_embeddings, axis=-0) if all_embeddings else tf.zeros((1, 1024))
  reshaped_concatenated_embeddings = tf.reshape(concatenated_embeddings, (-1, 1024))
  return reshaped_concatenated_embeddings

# BƯỚC 3.1:
# Lưu document embeddings vào file
def save_embeddings_to_file(embeddings, file_path=embeddings_file_path):
  # Convert TensorFlow tensor hoặc PyTorch tensor sang numpy nếu cần
  embeddings_np = np.array(embeddings)
  np.save(file_path, embeddings_np)
  print(f"Saved embeddings to {file_path}")

# Đọc document embeddings từ file
def load_embeddings_from_file(file_path="embeddings.npy"):
  if not os.path.exists(file_path):
    print(f"File {file_path} does not exist.")
    return None
  embeddings = np.load(file_path)
  print(f"Loaded embeddings from {file_path}")
  return embeddings

def get_or_generate_embeddings(texts, file_path=embeddings_file_path):
  # Kiểm tra xem file embeddings đã tồn tại chưa
  embeddings = load_embeddings_from_file(file_path)
  
  # Nếu embeddings chưa tồn tại, generate và save
  if embeddings is None:
    embeddings = generate_embeddings(texts)
    save_embeddings_to_file(embeddings, file_path)
  
  return embeddings

# BƯỚC 4:
# Xếp hạng các document liên quan nhất dựa trên độ tương sự cosin (cosine similarity)
# input: vector embedding của câu truy vấn và các văn bản bên ngoài (documents)
# output: mảng kết quả cosine similarity 
# và mảng chỉ số tài liệu (document index) đã được sắp xếp theo cos_sim giảm dần
def rank_documents(query_embedding, doc_embeddings):
  cos_similarities = cosine_similarity(query_embedding, doc_embeddings)
  ranked_indices = np.argsort(-cos_similarities, axis=1)
  return cos_similarities, ranked_indices

# BƯỚC 5:
# Lấy một số lượng k những document được xếp hạng cao nhất
def retrieve_top_k_documents(query_embedding, doc_embeddings, documents, k = k_count):
  cos_similarities, ranked_indices = rank_documents(query_embedding, doc_embeddings)
  top_k_indices = ranked_indices[0][:k]
  top_k_documents = [documents[idx] for idx in top_k_indices if idx < len(documents)]
  return top_k_documents

# BƯỚC 6:
# Xếp hạng lại danh sách những document đã được chọn ở bước 5
def re_rank_documents(query, top_k_documents):
  inputs = [f"{query} [SEP] {doc}" for doc in top_k_documents]
  tokenized_inputs = ranking_tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
  with torch.no_grad():
    outputs = ranking_model(**tokenized_inputs)

  scores = outputs.logits.squeeze().tolist()
  if not isinstance(scores, list):
    scores = [scores]

  ranked_docs = [doc for _, doc in sorted(zip(scores, top_k_documents), reverse=True)]
  print("\nRanked Documents:")
  for i, doc in enumerate(ranked_docs):
    print(f"Rank {i+1}: Score={scores[i]}, Document: {doc[:100]}...")
  return ranked_docs

# BƯỚC 7:
# Tạo câu trả lời từ câu truy xuất + những document từ bước 6
def generate_response(query, top_k_documents):
  if isinstance(query, list):
    query = ' '.join(query)
  
  # Chuyển từng phần tử trong top_k_documents thành chuỗi nếu nó là danh sách
  top_k_documents = [' '.join(doc) if isinstance(doc, list) else doc for doc in top_k_documents]
  
  # Nối các tài liệu lại với nhau thành một chuỗi
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

# BƯỚC 7.1:
# Loại bỏ "<token_id_x>" khỏi câu trả lời
def remove_extra_ids(text):
  pattern = r'<extra_id_\d+>'
  cleaned_text = re.sub(pattern, '', text)
  return cleaned_text.strip()

# BƯỚC 8:
# Khởi tạo chatbot
def initialize_chatbot():
  initialize_models()

  global documents
  documents = get_data_from_txt_files(external_document_dir)

  global segmented_documents, doc_embeddings
  segmented_documents = [segment_words(document) for document in documents]
  doc_embeddings = get_or_generate_embeddings(segmented_documents)

initialize_chatbot()

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
    segmented_user_message = segment_words(user_message)
    query_embedding = generate_embeddings([segmented_user_message])
    top_k_documents = retrieve_top_k_documents(query_embedding, doc_embeddings, documents, k=3)
    ranked_docs = re_rank_documents(user_message, top_k_documents)
    response = generate_response(user_message, ranked_docs)
    response = remove_extra_ids(response)

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
  app.run_server(debug=False)