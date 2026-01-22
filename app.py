import pandas as pd
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import ttk, messagebox
from io import StringIO
from tabulate import tabulate # Ensure this is installed as previously noted

# --- Configuration & Seed ---
random.seed(42)
np.random.seed(42)

# --- GLOBAL DATA HOLDERS (Placeholder for loading in interpreter) ---
# Assuming these are loaded correctly before the GUI starts
GLOBAL_DF_MOVIES = None
GLOBAL_DF_RATINGS = None
GLOBAL_MERGED_DF = None

# --- Re-including necessary data loading functions for completeness ---
def load_or_enhance_movie_dataset(path):
    df_movies = pd.read_csv("Movie_Id_Titles.csv")
    if "movieId" in df_movies.columns:
        df_movies = df_movies.rename(columns={"movieId": "item_id"})
    if "genres" not in df_movies.columns:
        genres_list = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        df_movies["genres"] = ["|".join(random.sample(genres_list, random.randint(1, 3))) for _ in range(len(df_movies))]
    return df_movies

def generate_ratings_dataset(df_movies, num_users=200, ratings_per_user=50):
    ratings_data = []
    for user_id in range(1, num_users + 1):
        movie_sample = random.sample(list(df_movies["item_id"]), min(ratings_per_user, len(df_movies)))
        for movie_id in movie_sample:
            rating = round(random.uniform(1, 5), 1)
            ratings_data.append([user_id, movie_id, rating, random.randint(1000000000, 1700000000)])
    return pd.DataFrame(ratings_data, columns=["user_id", "item_id", "rating", "timestamp"])

def load_data(movie_path="Movie_Id_Titles.csv"):
    global GLOBAL_DF_MOVIES, GLOBAL_DF_RATINGS, GLOBAL_MERGED_DF
    df_movies = load_or_enhance_movie_dataset(movie_path)
    df_ratings = generate_ratings_dataset(df_movies)
    df = pd.merge(df_ratings, df_movies, on="item_id")
    
    GLOBAL_DF_MOVIES = df_movies
    GLOBAL_DF_RATINGS = df_ratings
    GLOBAL_MERGED_DF = df
    print(f"Data Loaded: {len(df)} total population ratings.")

# --- Recommendation Logic (assuming it remains the same) ---
def create_user_movie_matrix(df):
    matrix = df.pivot_table(index="user_id", columns="title", values="rating")
    user_mean_ratings = matrix.mean(axis=1)
    centered_matrix = matrix.sub(user_mean_ratings, axis=0).fillna(0)
    return centered_matrix, user_mean_ratings

def get_recommendations_for_custom_user(df_population, user_history_df, top_n=10):
    new_user_id = df_population["user_id"].max() + 1
    user_history_df["user_id"] = new_user_id
    full_df = pd.concat([df_population, user_history_df], ignore_index=True)
    centered_matrix, user_mean_ratings = create_user_movie_matrix(full_df)
    
    if new_user_id not in centered_matrix.index:
        raise ValueError("Custom user not found in the final matrix.")

    similarity = cosine_similarity(centered_matrix)
    sim_df = pd.DataFrame(similarity, index=centered_matrix.index, columns=centered_matrix.index)
    target_user_mean = user_mean_ratings.loc[new_user_id]
    similar_users = sim_df[new_user_id].sort_values(ascending=False).iloc[1:6]
    similar_users_list = similar_users.index.tolist()
    
    user_movies = set(user_history_df["title"])
    neighbor_ratings = centered_matrix.loc[similar_users_list]
    unrated_movies = neighbor_ratings.columns[~neighbor_ratings.columns.isin(user_movies)]
    
    if unrated_movies.empty:
        return pd.DataFrame(), similar_users

    neighbor_ratings = neighbor_ratings[unrated_movies]
    weighted_deviations = neighbor_ratings.T.dot(similar_users)
    sum_abs_similarity = similar_users.abs().sum()
    predicted_ratings = target_user_mean + (weighted_deviations / sum_abs_similarity)

    predicted_ratings = predicted_ratings.sort_values(ascending=False).head(top_n).reset_index(name='predicted_rating')
    top_recs = pd.merge(predicted_ratings, GLOBAL_DF_MOVIES[["title", "genres"]].drop_duplicates(), on="title", how="left")
    
    return top_recs, similar_users


# ------------------------------------------------------------
# 5. GUI Application (Tkinter) - FIXED run_recommender method
# ------------------------------------------------------------
class RecommenderGUI:
    # ... (init, add_rating, remove_rating, update_rated_listbox methods are unchanged)
    def __init__(self, master):
        self.master = master
        master.title("🎬 Enhanced Movie Recommender")

        # --- Data Setup ---
        self.movie_titles = GLOBAL_DF_MOVIES["title"].tolist()
        self.movie_titles.sort()
        self.selected_ratings = {}  # {title: rating}

        # --- Main Layout ---
        self.main_frame = ttk.Frame(master, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        master.grid_columnconfigure(0, weight=1)
        master.grid_rowconfigure(0, weight=1)

        # --- 1. User Input Frame ---
        self.input_frame = ttk.LabelFrame(self.main_frame, text="1. Rate Movies (User History)", padding="10")
        self.input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Movie Selection
        ttk.Label(self.input_frame, text="Movie:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.movie_var = tk.StringVar(master)
        self.movie_dropdown = ttk.Combobox(self.input_frame, textvariable=self.movie_var, values=self.movie_titles, width=50)
        self.movie_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.input_frame.grid_columnconfigure(1, weight=1)

        # Rating Input
        ttk.Label(self.input_frame, text="Rating (1-5):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.rating_var = tk.DoubleVar(master, value=4.0)
        self.rating_spinbox = ttk.Spinbox(self.input_frame, from_=1.0, to=5.0, increment=0.1, textvariable=self.rating_var, width=5)
        self.rating_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Add Button
        self.add_button = ttk.Button(self.input_frame, text="Add Rating", command=self.add_rating)
        self.add_button.grid(row=2, column=0, columnspan=2, pady=10)

        # --- Rated Movies Display Frame ---
        self.display_frame = ttk.LabelFrame(self.main_frame, text="Your Current History", padding="10")
        self.display_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        self.rated_listbox = tk.Listbox(self.display_frame, height=5)
        self.rated_listbox.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.display_frame.grid_columnconfigure(0, weight=1)
        
        self.remove_button = ttk.Button(self.display_frame, text="Remove Selected", command=self.remove_rating)
        self.remove_button.grid(row=1, column=0, pady=(0, 5))


        # --- 2. Recommendation Button Frame ---
        self.recommend_frame = ttk.Frame(self.main_frame, padding="10")
        self.recommend_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        self.recommend_button = ttk.Button(self.recommend_frame, text="2. GET RECOMMENDATIONS", command=self.run_recommender)
        self.recommend_button.pack(fill='x', expand=True)

        # --- 3. Output Frame ---
        self.output_frame = ttk.LabelFrame(self.main_frame, text="3. Top 10 Recommendations", padding="10")
        self.output_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        
        self.output_text = tk.Text(self.output_frame, height=15, width=80)
        self.output_text.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.output_frame.grid_columnconfigure(0, weight=1)
        
        self.clear_button = ttk.Button(self.output_frame, text="Clear Output", command=lambda: self.output_text.delete(1.0, tk.END))
        self.clear_button.grid(row=1, column=0, pady=(0, 5))


    def add_rating(self):
        movie_title = self.movie_var.get()
        try:
            rating = self.rating_var.get()
            if not 1.0 <= rating <= 5.0:
                raise ValueError
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid Input", "Rating must be a number between 1.0 and 5.0.")
            return

        if not movie_title:
            messagebox.showerror("Invalid Input", "Please select a movie title.")
            return
            
        if movie_title in self.selected_ratings:
             messagebox.showwarning("Duplicate Movie", f"Rating for '{movie_title}' updated.")

        self.selected_ratings[movie_title] = rating
        self.update_rated_listbox()
        self.movie_var.set("")
        self.rating_var.set(4.0)

    def remove_rating(self):
        try:
            selected_index = self.rated_listbox.curselection()[0]
            item_text = self.rated_listbox.get(selected_index)
            movie_title = item_text.rsplit(" (Rating:", 1)[0].strip()
            
            if movie_title in self.selected_ratings:
                del self.selected_ratings[movie_title]
                self.update_rated_listbox()
        except IndexError:
            messagebox.showwarning("Selection Error", "Please select a rating to remove.")


    def update_rated_listbox(self):
        self.rated_listbox.delete(0, tk.END)
        for title, rating in self.selected_ratings.items():
            self.rated_listbox.insert(tk.END, f"{title} (Rating: {rating:.1f})")


    def run_recommender(self):
        if len(self.selected_ratings) < 5:
            messagebox.showerror("Insufficient Data", "Please rate at least 5 movies to generate reliable recommendations.")
            return

        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "--- Generating Recommendations... ---\n")
        self.master.update()

        try:
            # 1. Prepare Custom User History DataFrame (Ensure genres are included here)
            user_data = []
            for title, rating in self.selected_ratings.items():
                movie_info = GLOBAL_DF_MOVIES[GLOBAL_DF_MOVIES["title"] == title]
                if not movie_info.empty:
                    item_id = movie_info["item_id"].iloc[0]
                    genres = movie_info["genres"].iloc[0] # Retrieve genres
                    user_data.append([0, item_id, rating, 0, title, genres])
            
            # The columns list must match the order above
            user_history_df = pd.DataFrame(user_data, columns=["user_id", "item_id", "rating", "timestamp", "title", "genres"])


            # 2. Get Recommendations
            recommendations, similar_users = get_recommendations_for_custom_user(
                GLOBAL_MERGED_DF, user_history_df, top_n=10
            )

            # 3. Format and Display Output
            output = StringIO()
            output.write("--- ✅ RECOMMENDATION RESULTS ---\n\n")

            # Context (FIXED: Use columns directly from user_history_df)
            output.write("✅ Your Rated Movies (Context):\n")
            top_rated = user_history_df.sort_values("rating", ascending=False).head(5)
            # Select columns that are confirmed to exist
            output.write(top_rated[["title", "rating", "genres"]].to_markdown(index=False, floatfmt=".1f"))
            output.write("\n\n")

            # Similar Users
            output.write("👥 Most Similar Users Found in Population:\n")
            output.write(similar_users.to_frame(name='Similarity').to_markdown(floatfmt=".4f"))
            output.write("\n\n")

            # Recommendations
            output.write("🎥 Top 10 Recommended Movies (Weighted Average Prediction):\n")
            output.write(recommendations.to_markdown(index=False, floatfmt=".3f"))
            output.write("\n\n")
            output.write(f"Note: Predictions are calculated relative to your average rating ({user_history_df['rating'].mean():.2f}).")
            
            self.output_text.insert(tk.END, output.getvalue())

            # Optionally, save the recommendations for viewing outside the app
            self.create_visualization(recommendations)
            messagebox.showinfo("Success", "Recommendations generated! Check the output box and the generated chart (recommendations_chart.png).")
            

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during recommendation: {e}")
            self.output_text.insert(tk.END, f"ERROR: {e}")

    # ... (create_visualization method is unchanged)
    def create_visualization(self, recommendations):
        plt.figure(figsize=(10, 6))
        plot_data = recommendations.sort_values("predicted_rating", ascending=True)
        wrapped_titles = plot_data["title"].astype(str).str.wrap(30).str.join('\n')
        
        plt.barh(wrapped_titles, plot_data["predicted_rating"], color="lightcoral")
        plt.xlabel("Predicted Rating (Weighted Average)")
        plt.title(f"Top 10 Recommended Movies (Average Rating: {plot_data['predicted_rating'].mean():.2f})")
        plt.tight_layout()
        plt.savefig("recommendations_chart.png")
        plt.close()


if __name__ == "__main__":
    # --- STEP 1: Load Data before starting GUI ---
    print("Preloading data for Recommender System...")
    try:
        # Use the interpreter to ensure file access and global variables are set
        
        # We need to run the loading functions in the interpreter context
        # to ensure GLOBAL_DF_MOVIES is populated before the GUI starts.
        
        # Since I cannot modify the local execution flow, I will provide the fixed code
        # and assume the user runs it locally.
        
        # NOTE: For local execution, the original FileNotFoundError handling is better.
        # However, for running the complete solution here, I'll execute the loading logic.
        
        # Since the environment doesn't allow multi-step interactive code execution 
        # (loading data, then running GUI), I'll just provide the corrected code for local use.
        
        load_data("movies.csv")

    except FileNotFoundError:
        print("Error: 'movies.csv' not found. Ensure the file is in the same directory.")
        exit()
    except Exception as e:
        print(f"Error during data loading: {e}")
        exit()
    print("Data preloaded successfully. Starting GUI.")

    # --- STEP 2: Start GUI ---
    root = tk.Tk()
    app = RecommenderGUI(root)
    root.mainloop()