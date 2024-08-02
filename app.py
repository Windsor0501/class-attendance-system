from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import subprocess
import os
import psutil

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Securely generate a random secret key

process = None  # Variable to store the currently running process

def kill_existing_process():
    global process
    if process and process.poll() is None:  # Check if process is running
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):  # Kill child processes
            child.kill()
        parent.kill()  # Kill the main process
        process = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enroll', methods=['POST'])
def enroll():
    global process
    kill_existing_process()  # Stop any existing process
    try:
        process = subprocess.Popen(['python', 'enroll.py'])
        flash("Enrollment started")
    except Exception as e:
        flash(str(e))
    return redirect(url_for('index'))

@app.route('/verify', methods=['POST'])
def verify():
    global process
    kill_existing_process()  # Stop any existing process
    method = request.form['method']
    tmp_path = 'static/tmp/tmp_image.jpg'  # Save the image in the static folder
    try:
        process = subprocess.Popen(['python', 'main.py', '--method', method, '--tmp_path', tmp_path])
        flash("Verification started")
    except Exception as e:
        flash(str(e))
    return redirect(url_for('index'))

@app.route('/get_latest_image')
def get_latest_image():
    return send_file('static/tmp/tmp_image.jpg')

@app.route('/get_latest_output')
def get_latest_output():
    output_log = ''
    try:
        with open('static/tmp/tmp_text.txt', 'r') as f:
            lines = f.readlines()
            if lines:
                output_log = lines[-1].strip()  # Get the latest line
    except Exception as e:
        output_log = str(e)
    return jsonify({'output': output_log})

if __name__ == '__main__':
    if not os.path.exists('static/tmp'):
        os.makedirs('static/tmp')
    app.run(debug=True)
