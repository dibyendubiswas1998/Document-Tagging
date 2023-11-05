from src.document_tagging.pipeline.prediction import PredictionPipeline
from flask import Flask, render_template, request



app = Flask(__name__)

@app.route("/")
def Home():
    """
        Renders the "index.html" template for the root URL ("/") in a Flask application.

        Returns:
            str: The rendered "index.html" template.
    """
    return render_template('index.html')


@app.route("/prediction", methods=['POST', 'GET'])
def Prediction():
    """
        Handles the POST and GET requests for the "/prediction" route.
        
        Returns:
        - The rendered "index.html" template with the predicted tags as a parameter.
    """
    pp = PredictionPipeline()
    tags = None
    if request.method == 'POST':
        messages = request.form['messages']
        tags = pp.prediction(data=messages) # get all the tags
    return render_template("index.html", tags=tags)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)