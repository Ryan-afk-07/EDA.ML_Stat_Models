from flask import Flask, request, jsonify, render_template, render_template_string
from flask_socketio import SocketIO, emit

##all flask routes
def register_routes(app):
    @app.route('/')
    def home():
        return render_template('main.html')

    @app.route('/gen1/')
    def gen1():
        return render_template('gen1.html')