import PIL.Image
from flask import Flask, request, jsonify, render_template, render_template_string
from flask_socketio import SocketIO, emit
import numpy as np
import pandas as pd
from ML.ASCII_bot.models import PokemonChatBot
from ML.ASCII_bot.routes import register_routes
from openai import OpenAI

app = Flask(__name__)
app_routes = register_routes(app)
socketio = SocketIO(app_routes, cors_allowed_origins='*')

if __name__ == '__main__':
    pokebot = PokemonChatBot(api_key='YOUR_OPENAI_API_KEY_HERE')    
    
    socketio.run(app, debug=True)