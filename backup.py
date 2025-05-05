from flask import Flask, render_template, request, redirect, session, jsonify
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from fuzzywuzzy import process
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

audio_features = [
    'Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness',
    'Acousticness', 'Instrumentalness', 'Liveness', 'Valence',
    'Tempo', 'Duration_ms', 'Time Signature'
]
# Load the CSV (DataFrame)
df = pd.read_csv("cleaned_spotify.csv")

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load the KNN model
with open("knn_model.pkl", "rb") as f:
    knn = pickle.load(f)



scaled_features = scaler.fit_transform(df[audio_features])
knn.fit(scaled_features)

track_uri_to_index = pd.Series(df.index, index=df['Track URI']).drop_duplicates()
extract_features = ["Track Name",
    "Track URI",
    "Track Preview URL",
    "Duration_ms",
    "Explicit",
    "Artist Name(s)",
    "Artist URI(s)",
    "Album Name",
    "Album Image URL",
    "Album URI",
    "Album Release Date",
    "Popularity",
    "Danceability",
    "Energy",
    "Valence",
    "Tempo",
    "Label",
    "Copyrights"]


def get_recommendations(track_uri, n=10):
    idx = track_uri_to_index[track_uri]
    distances, indices = knn.kneighbors([scaled_features[idx]], n_neighbors=n+1)
    similar_songs = indices[0][1:]  # Exclude the queried song
    return df.iloc[similar_songs][extract_features].to_dict(orient='records')

def search_Song(song_name):
    # Perform fuzzy matching to find the closest match to the provided song name
    matched_song = process.extractOne(song_name, df['Track Name'])

    if(matched_song is None):
        return 0

    if(matched_song[1] < 80):
        return 0

    # Unpack the result correctly
    matched_song_name, score = matched_song[0], matched_song[1]
    print(matched_song_name)

    matched_song_data = df.loc[matched_song[2]]
    print(matched_song)
    return matched_song_data #returns id

@app.route('/recommend_search', methods=['POST'])
def recommendation_engine2():
    song_name = ""
    if request.method == 'POST':
        song_name = request.form['song_name']
    if(song_name == ""):
        return "Please enter a song name"

    song_data = search_Song(song_name)
    source_data = song_data[extract_features].to_dict()
    source_uri = song_data['Track URI']
    recommendations = get_recommendations(source_uri)
    return jsonify({
        'source': source_data,
        'recommendations': recommendations,
        'success': True
    })


@app.route('/recommendation', methods=['POST'])
def recommendation_engine():
    song_uri = ""
    if request.method == 'POST':
       song_uri = request.form['song_uri']
    if(song_uri == ""):
        return "Please enter a song name"

    # Check if the song_uri is valid
    source = df[df['Track URI'] == song_uri]
    if source.empty:
        return jsonify({
            'error': 'Invalid song URI',
            'success': False
        })

    source_uri = song_uri
    recommendations = get_recommendations(source_uri)
    recommendations.append(source.iloc[0][extract_features].to_dict())
    return jsonify({
        'recommendations': recommendations,
        'success': True
    })

@app.route('/trend', methods=['GET'])
def get_trending_songs():
  df_sorted = df.sort_values(by='Popularity', ascending=False)
  top_songs = df_sorted.head(10).to_dict(orient='records')  # ✅ Correct format
  return jsonify({
      'recommendations': top_songs,
      'success': True
  })

@app.route('/search', methods=['POST'])
def search_song():
    song_name = request.form['song_name']
    if song_name == "":
        return "Please enter a song name"
    matched_song = process.extract(song_name, df['Track Name'])
    print(matched_song)
    ids = [item[2] for item in matched_song]
    songs = df.iloc[ids][extract_features].to_dict(orient='records')  # ✅ Proper format
    return jsonify({
      'recommendations': songs,
      'success': True
    })


@app.route('/')
def hello():
    return 'Hello World!'
