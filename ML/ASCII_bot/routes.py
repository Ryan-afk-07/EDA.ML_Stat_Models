from flask import Flask, request, jsonify, render_template, render_template_string
from flask_socketio import SocketIO, emit
from models import PokemonChatBot, PokemonCardBot


bot = PokemonCardBot(api_key='your_openai_api_key_here')   


##all flask routes
def register_routes(app):
    @app.route('/')
    def home():
        return render_template('poke_list.html')

    @app.route('/chat')
    def pokechat():
        user_msg = request.json['message']
        reply = bot.get_response(user_msg)
        return jsonify({'reply': reply})