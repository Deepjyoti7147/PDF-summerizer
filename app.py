from flask import Flask, request, send_file, render_template
import fitz  # PyMuPDF
import nltk
import openai
from fpdf import FPDF

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
    return send_file(new_pdf, attachment_filename='output.pdf')

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

def get_responses_from_gpt(tokens, prompt):
    responses = []
    openai.api_key = 'YOUR API KEY'
    for token in tokens:
        response = openai.Completion.create(
            engine="davinci",
            prompt=f"{prompt}\n\n{token}",
            max_tokens=1500
        )
        responses.append(response.choices[0].text)
    return responses

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
