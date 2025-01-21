from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Read Excel file
    df = pd.read_excel(file)
    # You can process the Excel data here
    return "File uploaded successfully", 200

if __name__ == "__main__":
    app.run(debug=True)
