from flask import Flask, request
import json
from flask_cors import CORS
from thematic_modeling import ThematicModeller
from datetime import datetime

app = Flask(__name__)
CORS(app)
startTime = datetime.now()
tm = ThematicModeller(
    dict_path='./src/data/wiki_en_output_wordids.txt.bz2',
    model_path='./src/data/wiki_lda.pkl')


@app.route('/get_themes', methods=['POST'])
def get_themes():
    request_json = request.get_json(silent=True)
    themes = tm.get_text_thems(request_json['raw_text'])
    return(json.dumps({'status': 'ok', 'themes': themes}), 200)


@app.route('/get_theme_words', methods=['POST'])
def get_theme_words():
    request_json = request.get_json(silent=True)
    words = tm.get_theme_words(request_json['theme_number'])
    return(json.dumps({'status': 'ok', 'words': words}), 200)


@app.route('/status', methods=['GET'])
def get_status():
    currTime = datetime.now()
    response = 'Uptime: {}'.format(currTime - startTime)
    return(response)


def main():
    app.run(host='0.0.0.0', port=5001)


if __name__ == '__main__':
    main()
