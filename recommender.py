# recommender.py
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

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
