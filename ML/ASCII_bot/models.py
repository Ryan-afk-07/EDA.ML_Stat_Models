from flask import Flask, request, jsonify, render_template, render_template_string
from flask_socketio import SocketIO, emit
from openai import OpenAI


##all flask classes/models
class PokemonChatBot():
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def get_response(self, user_input):
        prompt = f"You are a knowledgeable Pokemon bot and Pokemon card expert. Answer the following question: {user_input}"
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content']

class PokemonCardBot():
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        