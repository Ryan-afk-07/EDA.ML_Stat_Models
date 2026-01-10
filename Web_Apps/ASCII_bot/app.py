import PIL.Image
from flask import Flask, request, jsonify, render_template, render_template_string
from flask_socketio import SocketIO, emit
import numpy as np
import pandas as pd
from models import PokemonChatBot, PokemonCardBot
from routes import register_routes
from openai import OpenAI

app = Flask(__name__)
app_routes = register_routes(app)

if __name__ == '__main__':
    app_routes.run(debug=True)