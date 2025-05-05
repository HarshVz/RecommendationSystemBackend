from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import pickle
import pandas as pd
import numpy as np
from fuzzywuzzy import process
from flask_cors import CORS
from recommender import MoodAwareRecommenderWithEnergy  # Import the class

app = Flask(__name__)
CORS(app)

# ---------------------------
# 1. Define the MoodAwareRecommenderWithEnergy class
# ---------------------------

# Load the dataframe (Spotify data)
with open('./models/spotify_dataframe.pkl', 'rb') as f:
    df = pickle.load(f)

# ---------------------------
# 2. Load the Pickle Files
# ---------------------------

# Load the preprocessor
with open('./models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)


# ---------------------------
# 2. Feature Engineering
# ---------------------------

num_features = ['Valence', 'Energy', 'Acousticness', 'Tempo',
                'Speechiness', 'Instrumentalness', 'Liveness']

df['emotional_intensity'] = df['Valence'] * df['Energy']
df['acoustic_energy'] = (1 - df['Acousticness']) * df['Energy']
num_features += ['emotional_intensity', 'acoustic_energy']

processed_features = preprocessor.fit_transform(df[num_features])



# Load the KMeans clustering model
with open('./models/kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

df['mood_cluster'] = kmeans.fit_predict(processed_features[:, :3])
# Energy type classification
ENERGY_THRESHOLD = 0.6
df['energy_type'] = np.where(df['Energy'] > ENERGY_THRESHOLD, 'high-energy', 'emotional')



class MoodAwareRecommenderWithEnergy:
    def __init__(self, n_neighbors=50, mood_weight=2.5):
        self.n_neighbors = n_neighbors
        self.mood_weight = mood_weight
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.scaler = StandardScaler()

    def fit(self, features, mood_features, energy_types):
        self.original_features = features
        self.mood_features = mood_features
        self.energy_types = energy_types
        self.knn.fit(self.scaler.fit_transform(features))

    def recommend(self, target_idx, energy_filter='same', n=10):
        distances, indices = self.knn.kneighbors(self.scaler.transform([self.original_features[target_idx]]))
        target_mood = self.mood_features[target_idx]
        target_energy_type = self.energy_types[target_idx]

        weighted_scores = []

        for i, idx in enumerate(indices[0]):
            if idx == target_idx:
                continue

            if energy_filter == 'same' and self.energy_types[idx] != target_energy_type:
                continue  # skip songs with different energy types

            content_sim = 1 - distances[0][i]
            mood_sim = 1 - distance.cosine(target_mood, self.mood_features[idx])
            weighted_score = content_sim + (self.mood_weight * mood_sim)
            weighted_scores.append((idx, weighted_score))

        sorted_indices = sorted(weighted_scores, key=lambda x: x[1], reverse=True)[:n]
        return [idx for idx, _ in sorted_indices]


mood_features = processed_features[:, :3]
energy_types = df['energy_type'].values

recommender = MoodAwareRecommenderWithEnergy()
recommender.fit(processed_features, mood_features, energy_types)

# ---------------------------
# 3. Define Extract Features
# ---------------------------

extract_features = [
    "Track Name", "Track URI", "Track Preview URL", "Duration_ms", "Explicit",
    "Artist Name(s)", "Artist URI(s)", "Album Name", "Album Image URL", "Album URI",
    "Album Release Date", "Popularity", "Danceability", "Energy", "Valence", "Tempo",
    "Label", "Copyrights"
]

# ---------------------------
# 4. Recommendation Function
# ---------------------------

def get_recommendations(track_identifier, n=10):
    if track_identifier in df['Track URI'].values:
        target_idx = df[df['Track URI'] == track_identifier].index[0]
    elif track_identifier in df['Track Name'].values:
        target_idx = df[df['Track Name'] == track_identifier].index[0]
    else:
        return pd.DataFrame(columns=df.columns)

    similar_indices = recommender.recommend(target_idx, energy_filter='same', n=n)

    recommendations = df.iloc[similar_indices].copy()
    target_values = df.iloc[target_idx][['Valence', 'Energy']]
    recommendations['valence_diff'] = recommendations['Valence'] - target_values['Valence']
    recommendations['energy_diff'] = recommendations['Energy'] - target_values['Energy']

    return recommendations[extract_features]

# ---------------------------
# 5. Song Search Function
# ---------------------------

def search_Song(song_name):
    matched_song = process.extractOne(song_name, df['Track Name'])

    if matched_song is None or matched_song[1] < 80:
        return None  # Changed to None for better handling of no matches

    matched_song_name, score, matched_song_idx = matched_song
    matched_song_data = df.iloc[matched_song_idx]
    return matched_song_data  # Returns the song data

# ---------------------------
# 6. Flask Routes
# ---------------------------

@app.route('/recommend_search', methods=['POST'])
def recommendation_engine2():
    song_name = request.form.get('song_name', "")

    if not song_name:
        return "Please enter a song name"

    song_data = search_Song(song_name)
    if song_data is None:
        return jsonify({'error': 'Song not found', 'success': False})

    source_data = song_data[extract_features].to_dict()
    source_uri = song_data['Track URI']
    recommendations = get_recommendations(source_uri, n=10)

    final_recommendations = recommendations.to_dict(orient='records')
    return jsonify({
        'source': source_data,
        'recommendations': final_recommendations,
        'success': True
    })

@app.route('/recommendation', methods=['POST'])
def recommendation_engine():
    song_uri = request.form.get('song_uri', "")

    if not song_uri:
        return "Please enter a song name"

    source = df[df['Track URI'] == song_uri]
    if source.empty:
        return jsonify({
            'error': 'Invalid song URI',
            'success': False
        })

    recommendations = get_recommendations(song_uri, 10)
    recommendations = recommendations.to_dict(orient='records')
    recommendations.append(source.iloc[0][extract_features].to_dict())
    return jsonify({
        'recommendations': recommendations,
        'success': True
    })

@app.route('/trend', methods=['GET'])
def get_trending_songs():
    df_sorted = df.sort_values(by='Popularity', ascending=False)
    top_songs = df_sorted.head(10).to_dict(orient='records')
    return jsonify({
        'recommendations': top_songs,
        'success': True
    })

@app.route('/search', methods=['POST'])
def search_song():
    song_name = request.form.get('song_name', "")

    if not song_name:
        return "Please enter a song name"

    matched_song = process.extract(song_name, df['Track Name'])
    ids = [item[2] for item in matched_song]
    songs = df.iloc[ids][extract_features].to_dict(orient='records')
    return jsonify({
        'recommendations': songs,
        'success': True
    })

@app.route('/')
def hello():
    return 'Hello World!'

if __name__ == '__main__':
    app.run()
