import requests
from bs4 import BeautifulSoup
import re
import spacy
import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Scraper Function: Extract and Process Text from Webpage
def extract_and_process_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        paragraphs = soup.find_all("p")
        if not paragraphs:
            return None, "No text found on the webpage."

        def clean_text(text):
            text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
            text = text.strip()  # Remove leading/trailing spaces
            text = re.sub(r'[^A-Za-z0-9.,!?\'" \n]', '', text)  # Remove non-alphanumeric chars
            return text

        cleaned_paragraphs = [clean_text(para.get_text()) for para in paragraphs]
        tokenized_paragraphs = [sent.text.strip() for para in cleaned_paragraphs for sent in nlp(para).sents]

        knowledgebase_text = "\n".join(tokenized_paragraphs)

        return knowledgebase_text, None
    except Exception as e:
        return None, f"Error fetching or processing webpage: {str(e)}"

# Load models
embedding_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
qa_model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
qa_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
qa_pipeline = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)

# Function to process knowledgebase into FAISS index
def process_knowledgebase(text):
    global knowledgebase_sentences, index
    knowledgebase_sentences = text.split("\n")
    
    # Generate embeddings
    knowledgebase_embeddings = embedding_model.encode(knowledgebase_sentences)
    
    # Build FAISS index
    index = faiss.IndexFlatL2(knowledgebase_embeddings.shape[1])
    index.add(knowledgebase_embeddings)

# Function to retrieve relevant passage
def get_relevant_passage(query, top_k=1):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(query_embedding, top_k)
    return [knowledgebase_sentences[idx] for idx in indices[0]]

# Function to generate answer using DistilBERT
def get_answer(query, context):
    result = qa_pipeline(question=query, context=context)
    return result['answer']

# UI function to handle URL input and scraping
def handle_url_input(url):
    text, error = extract_and_process_text(url)
    if error:
        return error, None
    process_knowledgebase(text)
    return "Website processed successfully. You can now ask questions.", None

# UI function for chatbot interaction
def chatbot_interface(query):
    if not knowledgebase_sentences:
        return "Please enter a website URL and process it first."
    
    relevant_passages = get_relevant_passage(query)
    answer = get_answer(query, " ".join(relevant_passages))
    return answer

# Gradio UI
with gr.Blocks() as iface:
    gr.Markdown("# 🌐 AI Chatbot with Scraped Knowledgebase")
    
    with gr.Row():
        url_input = gr.Textbox(label="Enter Website URL")
        url_button = gr.Button("Scrape Website")
    
    status_output = gr.Textbox(label="Status", interactive=False)
    
    url_button.click(handle_url_input, inputs=url_input, outputs=[status_output])

    gr.Markdown("## 💬 Ask a Question")
    query_input = gr.Textbox(label="Your Question")
    answer_output = gr.Textbox(label="Chatbot Answer", interactive=False)
    query_button = gr.Button("Get Answer")

    query_button.click(chatbot_interface, inputs=query_input, outputs=answer_output)

iface.launch()
