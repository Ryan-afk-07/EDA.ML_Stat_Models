import PIL.Image
from flask import Flask, request, jsonify, render_template, render_template_string
from flask_socketio import SocketIO, emit
import numpy as np
import pandas as pd
from openai import OpenAI

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

##all flask classes
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

##all flask routes
@app.route('/')
def home():
    return render_template('main.html')

@app.route('/gen1/')
def gen1():
    return render_template('gen1.html')



#setting ASCII characters that will be employed to the final ASCII art
ASCII_special = ["@", "#","$", "%", "?","*","+",";",":",",",".", "~", "!", "|", "/","<", ">", "=", "'", " ", "{", "}", "_", "^"]
ASCII_lower = ["a", "b", "c", "d", "e", "f", "g", "h", "i","j","k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
ASCII_upper = [lambda char: char.upper() for char in ASCII_lower]
ASCII_num = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
ASCII_fin = ASCII_special + ASCII_lower + ASCII_upper + ASCII_num

#resize image according to a new set width
def resize_image(image, new_width=100):
    width, height = image.size
    ratio = height / width
    new_height = int(new_width * ratio)
    resized_image = image.resize((new_width, new_height))
    return resized_image

#convert each pixel to grayscale
def grayify(image):
    grayscale_image = image.convert("L")
    return grayscale_image

#convert pixels to a string of ASCII characters
def pixels_to_ascii(image):
    pixels = image.getdata()
    characters = "".join([ASCII_fin[pixel // 25] for pixel in pixels])
    return (characters)

def ascii_convert(new_width=100):
    path = input("Enter image path")
    try:
        image = PIL.Image.open(path)
    except:
        print('Not a valid path. Please try again.')
    
    new_image_data = pixels_to_ascii(grayify(resize_image(image))) 
    pixel_count = len(new_image_data)
    ascii_image = "\n".join([new_image_data[index:(index+new_width)] for index in range(0, pixel_count, new_width)])

    print(ascii_image)
    #save the ASCII art to a text file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_image)   

pokebot = PokemonChatBot(api_key='YOUR_OPENAI_API_KEY_HERE')    

@socketio.on('user_messa')
def pokechat(data):
    reply = pokebot.get_response(data['message'])
    emit('bot_reply', {'reply': reply})

if __name__ == '__main__':
    socketio.run(app, debug=True)