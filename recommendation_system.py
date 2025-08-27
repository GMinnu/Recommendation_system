import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-activity ratings data (replace with your real data)
data = {
    'User': ['Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Carol', 'Carol', 'Carol', 'Dave', 'Dave'],
    'Activity': ['Hiking', 'Cooking', 'Reading', 'Hiking', 'Painting', 'Cooking', 'Reading', 'Dance', 'Reading', 'Dance'],
    'Rating': [5, 3, 4, 4, 5, 5, 3, 4, 2, 5]
}

df = pd.DataFrame(data)

# Create a user-item matrix
user_activity_matrix = df.pivot_table(index='User', columns='Activity', values='Rating').fillna(0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_activity_matrix)

# Convert to DataFrame for readability
user_sim_df = pd.DataFrame(user_similarity, index=user_activity_matrix.index, columns=user_activity_matrix.index)

print("User Similarity Matrix:\n", user_sim_df)

# Function to recommend activities to a user based on similar users
def recommend_activities(user, user_activity_matrix, user_sim_df, top_n=3):
    if user not in user_activity_matrix.index:
        print(f"User {user} not found in data.")
        return []

    # Get similarity scores for the user
    sim_scores = user_sim_df[user]

    # Get activities that user has not rated
    user_activities = user_activity_matrix.loc[user]
    unrated_activities = user_activities[user_activities == 0].index.tolist()

    recommendations = {}
    for activity in unrated_activities:
        # Weighted sum of ratings from similar users
        total_score = 0
        total_sim = 0
        for other_user in user_activity_matrix.index:
            if other_user == user:
                continue
            rating = user_activity_matrix.loc[other_user, activity]
            sim = sim_scores[other_user]
            total_score += rating * sim
            total_sim += sim
        if total_sim > 0:
            recommendations[activity] = total_score / total_sim

    # Sort recommendations by score
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

    return sorted_recs[:top_n]


# Example: Recommend activities for 'Bob'
recommended_for_bob = recommend_activities('Bob', user_activity_matrix, user_sim_df, top_n=3)
print("\nTop Recommendations for Bob:")
for activity, score in recommended_for_bob:
    print(f"{activity} (score: {score:.2f})")
