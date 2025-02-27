import google.generativeai as genai
import chromadb
import os
from dotenv import load_dotenv
from google.api_core import retry
import PyPDF2
load_dotenv()
class PDFReader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
    
    def __call__(self):
        with open(self.pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return [page.extract_text() for page in pdf_reader.pages]

class Embedding:
    def __init__(self, mode):
        self.document_mode = mode
    
    def __call__(self, input):
        embedding_task = "retrieval_document" if self.document_mode else "retrieval_query"
        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=input,
            task_type=embedding_task,
            request_options=retry_policy,
        )
        return response["embedding"]

def main():


    GOOGLE_API_KEY ="AIzaSyCRIlcoUu4P1xqTfXq4A4XPXDWoie7F3zg"
    genai.configure(api_key=GOOGLE_API_KEY)

    # Database setup
    DB_NAME = "QnA"
    client = chromadb.Client()
    db = client.get_or_create_collection(name=DB_NAME)

    # Check if database contains documents
    doc_count = db.count()
    if doc_count == 0:
        print("No documents found in the database. Please add your PDF.")
        pdf_path = input("Enter the path to your PDF: ").strip()
        pdf_reader = PDFReader(pdf_path)
        pdf_text = pdf_reader()
        
        # Add documents to the database
        for idx, page in enumerate(pdf_text):
            db.add(
                documents=[page],
                ids=[f"page-{idx}"]
            )
        print("Document added successfully. You can now ask questions.")

    print("Ask me anything! (Type 'exit' to quit)")
    while True:
        query = input("Your question: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Query the database for relevant passages
        result = db.query(query_texts=[query], n_results=1)
        if not result["documents"]:
            print("Sorry, I couldn't find an answer. Try rephrasing your question.")
            continue

        # Extract the most relevant passage
        relevant_passages = result["documents"][0]
        if not relevant_passages:
            print("No relevant passage found.")
            continue

        passage_oneline = " ".join(relevant_passages).replace("\n", " ")
        
        # Use the passage and query to generate a response
        prompt = f"Query: {query}\nRelevant Passage: {passage_oneline}"
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        # Extract plain text from the response
        answer = model.generate_content(prompt).parts[0].text
        print(f"Answer: {answer}")
        # print(f"Passage: {passage_oneline}")


if __name__ == "__main__":
    main() 