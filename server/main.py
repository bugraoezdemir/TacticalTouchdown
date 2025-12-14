from flask import Flask, jsonify, request
from flask_cors import CORS
from game_engine import Game

app = Flask(__name__)
CORS(app)

game = Game()

@app.route('/state', methods=['GET'])
def get_state():
    return jsonify(game.to_dict())

@app.route('/tick', methods=['POST'])
def tick():
    game.iterate()
    return jsonify(game.to_dict())

@app.route('/reset', methods=['POST'])
def reset():
    game.reset_positions()
    return jsonify(game.to_dict())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
