from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', title='Dynamic Title', content='Hello, this is default content!')

@app.route('/page1', methods=['GET', 'POST'])
def page1():
    if request.method == 'POST':
        # Process form data
        user_input = request.form['user_input']
        processed_result = process_data(user_input)
        return render_template('result.html', result=processed_result)
    
    return render_template('page1.html', title='Page 1')

@app.route('/page2', methods=['GET', 'POST'])
def page2():
    if request.method == 'POST':
        # Process form data
        user_input = request.form['user_input']
        processed_result = process_data(user_input)
        return render_template('result.html', result=processed_result)
    
    return render_template('page2.html', title='Page 2')

def process_data(data):
    # Your data processing logic here
    # For example, let's just return the reversed string
    return data[::-1]

if __name__ == '__main__':
    app.run(debug=True)
