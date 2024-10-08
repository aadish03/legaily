# Standard library imports
import io
import json
import logging
import os
from typing import List

# Third-party imports
from deep_translator import GoogleTranslator
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pydantic import BaseModel
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import cohere
from langchain_community.llms import Cohere

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure your Google API key is set in the environment
google_api_key = "AIzaSyAjU-G6ZALGIx3i5npDVr7PdBX7_Uxdw70"
if not google_api_key:
    logger.error("Google API key not found in environment variables")
    raise ValueError("Google API key not found in environment variables")

# Initialize Cohere client
cohere_api_key = "lM3tDW2rDBFpwslHmzGaDbeEhVjpRDp1SnzlzhzT"
if not cohere_api_key:
    logger.error("Cohere API key not found in environment variables")
    raise ValueError("Cohere API key not found in environment variables")

cohere_client = cohere.Client(cohere_api_key)


# Configure the Gemini model
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel('gemini-pro')

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NLP tools and download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the pre-trained sentence transformer model
model_name = 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens'
sentence_model = SentenceTransformer(model_name)

# Load your dataset for the cases
with open('indian_court_cases.json', 'r') as f:
    data = json.load(f)
cases = data['cases']

# Generate embeddings for the cases
case_texts = [f"{case['case_title']} {case['description']}" for case in cases]
case_embeddings = sentence_model.encode(case_texts, convert_to_tensor=True)

def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by tokenizing, lowercasing, removing non-alphanumeric tokens,
    removing stopwords, and lemmatizing.
    """
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Define the request model for querying cases
class QueryModel(BaseModel):
    query: str

@app.post("/ask")
async def ask(query: QueryModel):
    """
    Endpoint to find similar cases based on the user's query.
    """
    user_input = query.query
    if not user_input:
        raise HTTPException(status_code=400, detail="No query provided")

    preprocessed_input = preprocess_text(user_input)
    user_embedding = sentence_model.encode(preprocessed_input, convert_to_tensor=True)

    # Calculate similarity between user input and case embeddings
    similarities = util.pytorch_cos_sim(user_embedding, case_embeddings)[0]
    top_indices = similarities.topk(5).indices.tolist()

    # Retrieve top 5 similar cases
    similar_cases = [cases[i] for i in top_indices]

    return {"cases": similar_cases}

@app.post("/process_pdf/")
async def process_pdf(file: UploadFile = File(...), query: str = Form(...), translation_language: str = Form(None)):
    """
    Endpoint to process a PDF file, extract text, chunk it, and answer a query based on the content using Cohere LLM.
    Optionally translates the answer to the specified language.
    """
    try:
        # Read the uploaded file and extract text
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
        raw_text = ' '.join(page.extract_text() or '' for page in pdf_reader.pages)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", "!", "?"],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(raw_text)
        
        # Combine chunks, limiting to approximately 10,000 characters
        combined_chunks = ""
        for chunk in chunks:
            if len(combined_chunks) + len(chunk) > 10000:
                break
            combined_chunks += chunk + " "
        
        # Prepare prompt for Cohere
        prompt = f"""You are a lawyer assistant. Please answer the following question based on the given document content.

Document content: {combined_chunks.strip()}

Question: {query}

Please provide a detailed answer based on the given document."""

        # Generate response using Cohere
        response = cohere_client.generate(
            model='command',
            prompt=prompt,
            max_tokens=500,
            temperature=0.7,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE'
        )
        answer = response.generations[0].text

        # Handle translation if requested
        if translation_language:
            try:
                translated_answer = GoogleTranslator(source='auto', target=translation_language).translate(answer)
                return JSONResponse(content={"answer": answer, "translated_answer": translated_answer})
            except Exception as e:
                logger.error(f"Translation error: {e}")
                return JSONResponse(content={"answer": answer, "error": str(e)})

        return JSONResponse(content={"answer": answer})

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

class ChatbotRequest(BaseModel):
    user_message: str

class ChatbotResponse(BaseModel):
    lawyer_response: str
    potential_charges: List[str]
    rights: List[str]
    next_steps: List[str]

@app.post("/chatbot", response_model=ChatbotResponse)
async def chatbot(request: ChatbotRequest):
    """
    Endpoint for the AI lawyer chatbot. Provides legal advice, potential charges,
    rights, and next steps based on the user's situation.
    """
    user_message = request.user_message

    # Prepare context and prompt for Gemini
    context = """You are an AI lawyer assistant. Provide legal advice, potential charges, 
    rights, and next steps based on the user's situation. Be concise and factual."""
    
    prompt = f"""Context: {context}

User situation: {user_message}

Please provide:
1. A brief lawyer's response
2. A list of potential charges (if applicable)
3. A list of relevant rights
4. A list of recommended next steps

Format your response as follows:
Lawyer's response: [Your response here]
Potential charges:
- [Charge 1]
- [Charge 2]
Relevant rights:
- [Right 1]
- [Right 2]
Next steps:
- [Step 1]
- [Step 2]
"""

    # Generate response using Gemini
    response = model.generate_content(prompt)
    answer = response.text

    # Parse the response
    lawyer_response = ""
    potential_charges = []
    rights = []
    next_steps = []

    current_section = None
    for line in answer.split('\n'):
        line = line.strip()
        if line.startswith("Lawyer's response:"):
            current_section = "lawyer_response"
            lawyer_response = line.replace("Lawyer's response:", "").strip()
        elif line.startswith("Potential charges:"):
            current_section = "potential_charges"
        elif line.startswith("Relevant rights:"):
            current_section = "rights"
        elif line.startswith("Next steps:"):
            current_section = "next_steps"
        elif line.startswith("- "):
            if current_section == "potential_charges":
                potential_charges.append(line.strip("- "))
            elif current_section == "rights":
                rights.append(line.strip("- "))
            elif current_section == "next_steps":
                next_steps.append(line.strip("- "))

    return ChatbotResponse(
        lawyer_response=lawyer_response,
        potential_charges=potential_charges,
        rights=rights,
        next_steps=next_steps
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)