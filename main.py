from flask import Flask, render_template, request, redirect, url_for
import os
import dotenv
import lancedb
import logging
from langchain_cohere import CohereEmbeddings
from langchain_community.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argostranslate.package
import argostranslate.translate

# Configuration
dotenv.load_dotenv(".env")
DB_PATH = "/tmp/lancedb"
COHERE_MODEL_NAME = "multilingual-22-12"
LANGUAGE_ISO_CODES = {"English": "en", "Hindi": "hi", "Turkish": "tr", "French": "fr"}

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Documents and Embeddings
def initialize_documents_and_embeddings(input_file_path):
    file_extension = os.path.splitext(input_file_path)[1]
    loader = TextLoader(input_file_path) if file_extension == '.txt' else PyPDFLoader(input_file_path)
    documents = loader.load() if file_extension == '.txt' else loader.load_and_split()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = CohereEmbeddings(model=COHERE_MODEL_NAME)
    return texts, embeddings

def initialize_database(texts, embeddings):
    db = lancedb.connect(DB_PATH)
    db.create_table("multiling-rag", data=[{"vector": embeddings.embed_query("Hello World"), "text": "Hello World", "id": "1"}], mode="overwrite")
    return LanceDB.from_documents(texts, embeddings, connection=db)

# Translation
def translate_text(text, from_code, to_code):
    try:
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages))
        argostranslate.package.install_from_path(package_to_install.download())
        return argostranslate.translate.translate(text, from_code, to_code)
    except Exception as e:
        logger.error(f"Error in translate_text: {str(e)}")
        return "Translation error"

# Question Answering
def answer_question(question, input_language, output_language, db):
    try:
        input_lang_code = LANGUAGE_ISO_CODES[input_language]
        output_lang_code = LANGUAGE_ISO_CODES[output_language]
        
        question_in_english = translate_text(question, from_code=input_lang_code, to_code="en") if input_language != "English" else question
        qa = RetrievalQA.from_chain_type(llm=Cohere(model="command", temperature=0), chain_type="stuff", retriever=db.as_retriever(), chain_type_kwargs={"prompt": PromptTemplate(template="{context}\n\nQuestion: {question}", input_variables=["context", "question"])}, return_source_documents=True)
        
        answer = qa({"query": question_in_english})
        result_in_english = answer["result"].replace("\n", "").replace("Answer:", "")
        
        return translate_text(result_in_english, from_code="en", to_code=output_lang_code) if output_language != "English" else result_in_english
    except Exception as e:
        logger.error(f"Error in answer_question: {str(e)}")
        return "An error occurred while processing your question. Please try again."

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        question = request.form['question']
        input_language = request.form['input_language']
        output_language = request.form['output_language']

        # Save the uploaded file
        if file and file.filename.endswith('.pdf'):
            file_path = os.path.join("/tmp", file.filename)  # Save to a temporary directory
            file.save(file_path)

            # Initialize documents and embeddings from the uploaded file
            texts, embeddings = initialize_documents_and_embeddings(file_path)
            db = initialize_database(texts, embeddings)

            answer = answer_question(question, input_language, output_language, db)
            return render_template('index.html', answer=answer, question=question, input_language=input_language, output_language=output_language)
    
    return render_template('index.html', answer=None)

if __name__ == "__main__":
    app.run(debug=True)
