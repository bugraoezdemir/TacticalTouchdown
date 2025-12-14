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

@app.route('/tactics', methods=['GET'])
def get_tactics():
    return jsonify(game.get_tactics())

@app.route('/tactics', methods=['POST'])
def set_tactics():
    data = request.get_json() or {}
    game.set_tactics(
        formation=data.get('formation'),
        mentality=data.get('mentality'),
        dribble_frequency=data.get('dribbleFrequency'),
        shoot_frequency=data.get('shootFrequency')
    )
    return jsonify(game.get_tactics())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
