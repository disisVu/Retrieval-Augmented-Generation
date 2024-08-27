import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import argparse
import warnings
import logging

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
warnings.filterwarnings("ignore", message=".*Setting `pad_token_id`.*")

# Suppress transformer logs
logging.getLogger("transformers").setLevel(logging.ERROR)

# Set environment variable for OpenMP runtime issue (temporary fix)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load the encoder model and tokenizer (for query processing)
encoder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
encoder_model = AutoModel.from_pretrained(encoder_model_name)

# Load the generation model (e.g., GPT-2 or any other LLM)
generator_model_name = "gpt2"
generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
generator_model = AutoModelForCausalLM.from_pretrained(generator_model_name)

# Function to load text files from a folder
def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Only process .txt files
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

# Function to encode documents into embeddings
def encode_documents(documents):
    document_embeddings = []
    for doc in documents:
        inputs = tokenizer(doc, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = encoder_model(**inputs)
        doc_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Pooling
        document_embeddings.append(doc_embedding)
    return np.array(document_embeddings)

# Normalize embeddings
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

# Load and encode documents
folder_path = "D:\\Vu\\23-24-3\\RAG\\demo\\external"
documents = load_documents_from_folder(folder_path)
document_embeddings = encode_documents(documents)
normalized_document_embeddings = normalize_embeddings(document_embeddings)

# Create and populate the FAISS index with cosine similarity
dimension = normalized_document_embeddings.shape[1]  # Dimensionality of the embeddings
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(normalized_document_embeddings)

# Store the document texts for retrieval later
document_store = documents

# Step 1: Query Processing
def encode_query(query):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = encoder_model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Pooling
    return query_embedding

# Normalize query embedding
def normalize_query_embedding(query_embedding):
    norm = np.linalg.norm(query_embedding)
    return query_embedding / norm

# Step 2: Document Retrieval
def retrieve_documents(query_embedding, top_k=5):
    normalized_query_embedding = normalize_query_embedding(query_embedding)
    distances, indices = faiss_index.search(normalized_query_embedding.reshape(1, -1), top_k)
    retrieved_docs = [document_store[idx] for idx in indices[0]]
    return retrieved_docs

# Step 3: Response Generation
def generate_response(query, retrieved_docs):
    # Ensure documents are unique and not excessively long
    unique_docs = list(set(retrieved_docs))

    # Format the combined text properly
    combined_text = query + "\n\n" + "\n\n".join(unique_docs)

    # Prepare the input text for the generation model
    inputs = generator_tokenizer(combined_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the response
    outputs = generator_model.generate(
        **inputs, 
        max_length=150, 
        num_return_sequences=1,  # Ensure only one response is generated
        no_repeat_ngram_size=2,  # Avoid repeating n-grams
        pad_token_id=generator_tokenizer.eos_token_id  # Handle padding correctly
    )

    # Decode and return the response
    response = generator_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Ensure the response ends properly and remove any unwanted dots or characters
    response = response.replace(" .", ".").replace(" ,", ",")
    
    return response



# Step 4: Delivery - Putting it all together
def rag_pipeline(query):
    query_embedding = encode_query(query)
    retrieved_docs = retrieve_documents(query_embedding)
    response = generate_response(query, retrieved_docs)
    return response

# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument("query", type=str, help="The query to process")
    args = parser.parse_args()
    
    query = args.query
    response = rag_pipeline(query)
    print(response)

if __name__ == "__main__":
    main()
