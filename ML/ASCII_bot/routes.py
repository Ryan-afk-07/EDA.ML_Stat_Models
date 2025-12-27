from flask import Flask, request, jsonify, render_template, render_template_string
from flask_socketio import SocketIO, emit
from models import PokemonChatBot, PokemonCardBot

bot = PokemonCardBot(api_key='YOUR_OPENAI_API_KEY_HERE')


##all flask routes
def register_routes(app):
    @app.route('/')
    def home():
        return render_template('poke_list.html')

    @app.route('/chat', methods=['POST'])
    def pokechat(): 
        data = request.get_json()
        user_msg = data.get('message', '')

        reply = bot.get_response(user_msg)
        return jsonify({'reply': reply})
    
    return app