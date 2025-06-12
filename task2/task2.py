import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("imdb_top_1000.csv")

df['combined_features'] = df['Director'] + ' ' + df['Genre'] + ' ' + df['Overview']

vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(df['combined_features'])

cosine_sim = cosine_similarity(feature_matrix)

def recommend_movies_by_director_only(director, df):
    filtered_df = df[df['Director'].str.lower() == director.lower()]
    if filtered_df.empty:
        return f"❌ Director '{director}' not found in the dataset."
    
    recommended_titles = filtered_df['Series_Title'].tolist()
    return recommended_titles

print(" Movie Recommender (Based on Director)")
print("Try directors like: Christopher Nolan, James Cameron, Steven Spielberg, Martin Scorsese\n")
user_director = input("Enter director's name: ")

results = recommend_movies_by_director_only(user_director, df)

print("\nRecommended Movies:") 
if isinstance(results, list):
    for i, movie in enumerate(results, 1):
        print(f"{i}. {movie}")
else:
    print(results)
