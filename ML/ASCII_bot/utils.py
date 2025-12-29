import PIL.Image
import numpy as np
import pandas as pd
from flask_socketio import SocketIO, emit
import requests #for HTTP requests
from bs4 import BeautifulSoup #for web scraping/HTML parsing
import time #for time delays during scraping
from datetime import datetime #for timestamping
from zoneinfo import ZoneInfo #for timezone handling
import json #for JSON handling
import os #for file system operations

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


#function to web scrape for top 5 cards for a given pokemon
def scrape_top_cards(pokemon_name):
    
    base_url = f"https://www.trollandtoad.com"
    url = f"{base_url}/search?q={pokemon_name.lower()}"
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/58.0.3029.110'),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",}
    cards = []
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all("a", href=True)[:5]

        for r in results:
            cards.append(r.get_text(strip=True))

        print("First 5 raw results:", cards)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []
    
    return cards