from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel
import os
import io
import json
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure your OpenAI API key is set in the environment
openai_api_key = "sk-proj-AeDqnx5rM0jGomB9L7RcT3BlbkFJ4xg3ajmi6zuIamGsMwQR"
if not openai_api_key:
    logger.error("OpenAI API key not found in environment variables")
    raise ValueError("OpenAI API key not found in environment variables")

os.environ["OPENAI_API_KEY"] = openai_api_key

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to match your frontend's URL in production
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

# NLP preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization and lowercasing
    tokens = [word for word in tokens if word.isalnum()]  # Remove non-alphanumeric tokens
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

# Define the request model for querying cases
class QueryModel(BaseModel):
    query: str

@app.post("/ask")
async def ask(query: QueryModel):
    user_input = query.query
    if not user_input:
        raise HTTPException(status_code=400, detail="No query provided")

    # Preprocess the user input
    preprocessed_input = preprocess_text(user_input)

    # Generate embedding for the user's query
    user_embedding = sentence_model.encode(preprocessed_input, convert_to_tensor=True)

    # Calculate similarity between user input and case embeddings
    similarities = util.pytorch_cos_sim(user_embedding, case_embeddings)[0]
    top_indices = similarities.topk(5).indices.tolist()

    # Retrieve top 5 similar cases
    similar_cases = [cases[i] for i in top_indices]

    return {"cases": similar_cases}

@app.post("/process_pdf/")
async def process_pdf(file: UploadFile = File(...), query: str = Form(...), translation_language: str = Form(None)):
    try:
        # Read the uploaded file
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
        
        # Extract text from the PDF
        raw_text = ''
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", "!", "?"],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_text(raw_text)

        # Generate embeddings and perform similarity search
        embeddings = OpenAIEmbeddings()
        document_search = FAISS.from_texts(texts, embeddings)
        docs = document_search.similarity_search(query)
        
        # Load QA chain and get answer
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        context = "You are a lawyer and provide assistance with legal questions."
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nPlease provide a detailed answer based on the given documents."
        answer = chain.run(input_documents=docs, question=prompt)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
