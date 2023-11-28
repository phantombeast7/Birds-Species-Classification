from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
import pandas as pd
from googlesearch import search
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)

# Load the ensemble model during application startup
model_path = 'ensemble_4model_final.h5'
birds_df = pd.read_csv('birds.csv')

# Load the model once globally
model = load_model(model_path)


# Function to preprocess an individual image
def preprocess_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), -1)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Function to calculate the offset based on class ID
def calculate_offset(class_id):
    if class_id <= 15:
        return 0
    elif class_id <= 72:
        return 1
    elif class_id <= 98:
        return 2
    elif class_id <= 189:
        return 3
    elif class_id <= 261:
        return 4
    elif class_id <= 314:
        return 5
    elif class_id <= 358:
        return 6
    elif class_id <= 396:
        return 7
    elif class_id <= 444:
        return 8
    elif class_id <= 510:
        return 9
    else:
        return 10


# Function to perform Google Search
def google_search(query):
    try:
        # Include both common and scientific names in the search query
        search_query = f"{query} bird {query.split()[0]}"
        search_results = list(search(search_query, num=1, stop=1))

        if search_results:
            first_result = search_results[0]
            title = first_result
            link = first_result
            return [{'title': title, 'link': link}]
        else:
            return []
    except Exception as e:
        print(f"Error in Google Search: {e}")
        return []

def get_wikipedia_info(link):
    try:
        response = requests.get(link)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract image URL
        img_tag = soup.find('meta', {'property': 'og:image'})
        image_url = img_tag['content'] if img_tag else None

        # Extract the first 3 paragraphs as a summary
        paragraphs = soup.find_all('p')[:3]
        summary = ' '.join([para.text.strip() for para in paragraphs])  # Extracting the first 3 paragraphs as a summary

        # Remove unwanted parts from the summary
        summary = summary.replace("Sign in to see your badges", "")
        summary = summary.replace("Sign in to see your stats", "")
        summary = summary.strip()

        return {'summary': summary, 'image_url': image_url}
    except Exception as e:
        print(f"Error getting Wikipedia information: {e}")
        return {'summary': "Wikipedia summary not available.", 'image_url': None}




# Define a route for the root URL
@app.route('/')
def index():
    return render_template('index.html')


# Define a route for the upload endpoint
@app.route('/upload', methods=['POST'])
def upload():
    preprocessed_image = preprocess_image(request.files['image'])
    print("Image processed successfully")

    prediction = model.predict([preprocessed_image] * 4)
    predicted_id = np.argmax(prediction)
    offset = calculate_offset(predicted_id)
    predicted_id_with_offset = predicted_id - offset

    if predicted_id_with_offset in range(len(birds_df)):
        predicted_species = birds_df.loc[birds_df['class id'] == predicted_id_with_offset, 'scientific name'].values[0]

        google_results = google_search(predicted_species)
        wiki_link = google_results[0]['link'] if google_results else None

        if wiki_link:
            wiki_info = get_wikipedia_info(wiki_link)
            print(f"\nBird Info: {wiki_info['summary']}")
            print(f"\nImage URL: {wiki_info['image_url']}")
        else:
            wiki_info = {'summary': "Wikipedia link not available.", 'image_url': None}
            print("No Wikipedia link available.")

        # Combine prediction result, Google Search information, and Wikipedia summary
        result = f'Predicted Bird Species: {predicted_species}'
        prediction = f'Bird Info: {wiki_info["summary"]}'
        image_url = wiki_info['image_url']
        print(f"Image URL passed to template: {image_url}")

        # Pass all variables to the template
        return render_template('index.html', result=result, prediction=prediction, image_url=image_url)

    else:
        return 'Invalid predicted class ID'


if __name__ == '__main__':
    app.run(debug=False)
