import os
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import openai
from flask import Flask, request, render_template

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Step 2: Chunk the text into smaller parts
def chunk_text(text, chunk_size=500):
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Step 3: Generate embeddings for the chunks using a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

# Step 4: Store the embeddings in FAISS for fast similarity search
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    index.add(embeddings)
    return index

# Step 5: Convert user query to embedding
def query_to_embedding(query):
    return model.encode([query])[0]

# Step 6: Search for the most relevant chunks using FAISS
def search_in_faiss(query_embedding, index, chunks, top_k=3):
    distances, indices = index.search(np.array([query_embedding]), top_k)
    results = [chunks[i] for i in indices[0]]
    return results

# Step 7: Generate a response using OpenAI GPT
openai.api_key = "your-openai-api-key"  # Replace with your OpenAI API key

def generate_response(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"Answer the following question based on the context:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = openai.Completion.create(
        engine="text-davinci-003",  # or "gpt-4" for more accurate results
        prompt=prompt,
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Step 8: Main function to bring it all together
def chat_with_pdf(pdf_path, query):
    # Step 1: Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Chunk the extracted text
    chunks = chunk_text(text)
    
    # Step 3: Generate embeddings for chunks
    embeddings = generate_embeddings(chunks)
    
    # Step 4: Create a FAISS index for fast retrieval
    index = create_faiss_index(np.array(embeddings))
    
    # Step 5: Convert user query into embedding
    query_embedding = query_to_embedding(query)
    
    # Step 6: Search the FAISS index for the most relevant chunks
    relevant_chunks = search_in_faiss(query_embedding, index, chunks)
    
    # Step 7: Generate response based on the relevant chunks using GPT
    response = generate_response(query, relevant_chunks)
    
    return response

# Flask web application
app = Flask(_name_)

# Route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle PDF upload and query submission
@app.route('/ask', methods=['POST'])
def ask():
    if request.method == 'POST':
        pdf_file = request.files['pdf']
        query = request.form['query']

        # Save the uploaded PDF to a temporary file
        pdf_path = os.path.join("uploads", pdf_file.filename)
        pdf_file.save(pdf_path)

        # Call the RAG pipeline to get the response
        response = chat_with_pdf(pdf_path, query)

        # Return the response to the user
        return render_template('index.html', response=response)

if _name_ == '_main_':
    # Ensure that the uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)