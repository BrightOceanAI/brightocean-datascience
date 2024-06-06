import json
import logging
import werkzeug.exceptions
from waitress import serve
from flask_cors import CORS
from flask import Flask, request, Response, jsonify

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.app_context().push()
logging.getLogger('flask_cors').level = logging.DEBUG

@app.route('/prediction', methods=['POST'])
def health_prediction():

    if request.method == 'POST':
        try:

            data = json.loads(request.data)

            return jsonify({
                'Saude': 
            })

    return ''


# Error Handling
@app.errorhandler(werkzeug.exceptions.BadRequest)
def handleBadRequest(error):
    return jsonify({'error': 'Má requisição'}), 400


# Error Handling
@app.errorhandler(werkzeug.exceptions.MethodNotAllowed)
def handleMethodNotAllowed(error):
    return jsonify({'error': 'Método não permitido'}), 405


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5001, threads=15, url_prefix="/coral-ai")
