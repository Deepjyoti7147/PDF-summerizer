from flask import Flask, request, send_file, render_template
import fitz  # PyMuPDF
import nltk
import openai
from fpdf import FPDF
import time
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    prompt = request.form['prompt']
    pdf_text = extract_text_from_pdf(file)
    tokens = tokenize_text(pdf_text)
    responses = get_responses_from_gpt(tokens, prompt)
    new_pdf = create_pdf_from_responses(responses)
    return send_file(new_pdf, download_name='output.pdf', as_attachment=True)

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def tokenize_text(text, token_size=500):
    tokens = nltk.sent_tokenize(text)
    chunks = [' '.join(tokens[i:i + token_size]) for i in range(0, len(tokens), token_size)]
    return chunks

openai.api_key = 'YOUR_API_KEY'
responses = []
def get_responses_from_gpt(tokens, prompt):
    
    max_retries = 5  # Maximum number of retries for rate limit errors
    
    for token in tokens:
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = openai.Completion.create(
                    engine="gpt-3.5-turbo",
                    prompt=f"{prompt}\n\n{token}",
                    max_tokens=150
                )
                responses.append(response.choices[0].text.strip())
                break  # Exit the retry loop on success
            except openai.error.RateLimitError:
                retry_count += 1
                wait_time = 2 ** retry_count  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except openai.error.OpenAIError as e:
                print(f"An error occurred: {e}")
                break  # Exit on other API errors
        else:
            print(f"Failed to get response for token: {token} after {max_retries} retries")
    
    return responses

print(responses)

def create_pdf_from_responses(responses):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for response in responses:
        pdf.multi_cell(0, 10, response)
    pdf_output = "output.pdf"
    pdf.output(pdf_output)
    return pdf_output

if __name__ == '__main__':
    app.run(debug=True)
