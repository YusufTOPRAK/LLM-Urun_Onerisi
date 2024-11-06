from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open('tfidf_vectorizer_3.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('product_prediction_model_3.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_content = request.form['content']
    user_rating = float(request.form['rating'])

    user_attributes = request.form['attributes'] 
    user_company = request.form['company']


    user_content_lower = user_content.lower()
    user_attributes_lower = user_attributes.lower()
    user_company_lower = user_company.lower()


    user_price_range = request.form['price']
    user_price_min, user_price_max = map(float, user_price_range.split('-'))  

    user_price_avg = (user_price_min + user_price_max) / 2


  
    new_data = {
        'content': [user_content_lower],
        'rating': [user_rating],
        'price(in $)': [user_price_avg],
        'product_attributes': [user_attributes_lower],
        'company': [user_company_lower]
    }
    
    new_data_df = pd.DataFrame(new_data)

    user_input_transformed = preprocessor.transform(new_data_df)

    predicted_product = model.predict(user_input_transformed)

    return jsonify(predicted_product=predicted_product[0])

if __name__ == "__main__":
    app.run(debug=True)
