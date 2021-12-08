from flask import Flask, json, request,jsonify
from process import run 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

app = Flask(__name__, template_folder="templates")

@app.route("/", methods = ['GET'])
def helloworld():
    return '''<h1>Hotmail Captcha Solver <2 same items in dice></h1>'''

@app.route("/api/captcha-solver", methods=['POST'])
def predict():
    if request.json:
        # predict image
        result = run(request.json)
        print('Success!')
        return jsonify({'result':result})
    print('Fail!')
    return jsonify({'result':'fail'})


@app.route("/shutdown", methods=['GET'])
def shutdown():
    '''Shutdown the server'''
    shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if shutdown_func is None:
        raise RuntimeError('Not running werkzeug')
    shutdown_func()
    return "Shutting down..."

if __name__ == "__main__":
    app.run(debug=True)