from flask import Flask, render_template, request
import xgboost as xgb
import trafilatura as trf
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load your XGBoost model
model = xgb.XGBClassifier()
model.load_model('model/xlm-roberta_xgboost_model.json')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    transformer = SentenceTransformer('aditeyabaral/sentencetransformer-xlm-roberta-base')
    prediction_result = "Phishing" # or it will be "Legitimate"
    if 'htmlFile' not in request.files:
        return "No file part"
    
    file = request.files['htmlFile']

    if file.filename == '':
        return "No selected file"

    # Save the uploaded file to a temporary location
    file_path = 'test/' + file.filename
    file.save(file_path)

    # START of the business logic here

    # Perform prediction using the file_path with your XGBoost model

    try:
        # Parse HTML content with trafilatura
        try:
            with open(file_path, 'r', encoding='utf-8') as saved_file:
                saved_content = saved_file.read()
        except:
            with open(file_path, 'r', encoding='windows-1256') as saved_file:
                saved_content = saved_file.read()
        # txt_content = trf.html2txt(file)
        html_content = trf.extract(saved_content)
        print("--------------------HTML CONTENT---------------------")
        print(html_content)

        # Get embeddings of the parsed HTML content
        embeddings = transformer.encode(html_content)
        print("--------------------EMBEDDINGS---------------------")
        embeddings = embeddings.reshape(1, -1)
        print(embeddings)


        prediction = model.predict(embeddings)
        print("--------------------PREDICTION---------------------")
        print(prediction)
        print(type(prediction))
        print(prediction.shape)
        if prediction == 0:
            prediction_result = "Legitimate"
        elif prediction == 1:
            prediction_result = "Phishing"

    except Exception as e:
        print(f"Error: {e}")
        return "Error occurred during prediction"

    return f"{file_path} is {prediction_result}"

if __name__ == '__main__':
    app.run(debug=True, port=5050)
