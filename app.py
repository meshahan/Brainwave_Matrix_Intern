from flask import Flask, request, render_template
import pickle

# Load the vectorizer and model
vector = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("fnds_model.pkl", 'rb'))

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        # Extract the news headline from the form
        news_headline = request.form['headline']
        
        # Transform the input data using the loaded TfidfVectorizer
        data = vector.transform([news_headline])
        
        # Make a prediction using the loaded model
        prediction = model.predict(data)[0]
        
        # Return the result on the same page
        return render_template("index.html", prediction=prediction, headline=news_headline)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
