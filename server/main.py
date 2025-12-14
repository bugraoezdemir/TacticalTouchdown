from flask import Flask, jsonify, request
from flask_cors import CORS
from game_engine import Game
import sys
import os

app = Flask(__name__)
CORS(app)

game = Game()

@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify(game.to_dict())

@app.route('/api/tick', methods=['POST'])
def tick():
    game.iterate()
    return jsonify(game.to_dict())

@app.route('/api/reset', methods=['POST'])
def reset():
    game.reset_positions()
    return jsonify(game.to_dict())

if __name__ == '__main__':
    # Use 5001 to avoid conflict with Vite on 5000 (proxied)
    app.run(host='0.0.0.0', port=5001)
