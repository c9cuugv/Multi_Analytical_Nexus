from flask import Flask, render_template, request
from Analyze import analyze
from new1 import generate_summary
from Analyze2 import non

app = Flask(__name__, template_folder='templates')

def process_text(text, analysis_type='analyze'):
    # Placeholder for your processing logic
    if analysis_type == 'analyze':
        return analyze(text)
    elif analysis_type == 'summarize':
        return generate_summary(text)
    elif  analysis_type == 'Email_Fraud_Analyzer':
        return non(text)
    else:
        return "Invalid analysis type"

@app.route('/analyze', methods=['GET', 'POST'])
def process_input_analyze():
    text_input = None
    result = None

    if request.method == 'POST':
        try:
            # Get text input from the form
            text_input = request.form['textInput']

            # Process the text using your machine learning model or function
            result = process_text(text_input, analysis_type='analyze')

        except Exception as e:
            # Handle any errors that might occur during processing
            result = f"Error: {str(e)}"

    return render_template('index.html', text_input=text_input, result=result)

@app.route('/summarize', methods=['GET', 'POST'])
def process_input_summarize():
    text_input = None
    result = None

    if request.method == 'POST':
        try:
            # Get text input from the form
            text_input = request.form['textInput']

            # Process the text using your machine learning model or function
            result = process_text(text_input, analysis_type='summarize')

        except Exception as e:
            # Handle any errors that might occur during processing
            result = f"Error: {str(e)}"

    return render_template('index1.html', text_input=text_input, result=result)

@app.route('/Email_Fraud_Analyzer', methods=['GET', 'POST'])
def process_input_Email_analyzer():
    text_input = None
    result = None

    if request.method == 'POST':
        try:
            # Get text input from the form
            text_input = request.form['textInput']

            # Process the text using your machine learning model or function
            result = process_text(text_input, analysis_type='Email_Fraud_Analyzer')

        except Exception as e:
            # Handle any errors that might occur during processing
            result = f"Error: {str(e)}"

    return render_template('index2.html', text_input=text_input, result=result)

if __name__ == "__main__":
    app.run(debug=True)




# @app.route('/', methods=['GET', 'POST'])
# def process_input():
#     result = None

#     # if request.method == 'POST':
#     #     try:
#     #         # Get text input from the form
#     #         text_input = request.form['textInput']

#     #         # Process the text using your machine learning model or function
#     #         result = analyze(text_input)

#     #     except Exception as e:
#     #         # Handle any errors that might occur during processing
#     #         result = f"Error: {str(e)}"

#     return render_template('index1.html', result=result)

# @app.route('/', methods=['GET', 'POST'])
# def process_input():
#     result = None

#     # if request.method == 'POST':
#     #     try:
#     #         # Get text input from the form
#     #         text_input = request.form['textInput']

#     #         # Process the text using your machine learning model or function
#     #         result = analyze(text_input)

#     #     except Exception as e:
#     #         # Handle any errors that might occur during processing
#     #         result = f"Error: {str(e)}"

#     return render_template('page1.html', result=result)

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)

# if __name__ == '__main__':
#     app.run(debug=True, port=5002)

# # app2/app.py
# if __name__ == '__main__':
#     app.run(debug=True, port=5002)

# # app3/app.py
# if __name__ == '__main__':
#     app.run(debug=True, port=5003)