"""
Sistema de Recomendación de Películas
Utilizando análisis de similitud de contenido basado en características de películas
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
import warnings
warnings.filterwarnings('ignore')


class MovieRecommender:
    """Clase principal para generar recomendaciones de películas"""
    
    def __init__(self, data_dir='data/'):
        """
        Inicializa el sistema de recomendación
        
        Args:
            data_dir (str): Ruta del directorio con los archivos CSV
        """
        self.data_dir = data_dir
        self.movies = None
        self.cosine_sim = None
        self.load_and_prepare_data()
    
    def load_and_clean_data(self):
        """
        Carga y limpia los datos de los archivos CSV
        
        Returns:
            pd.DataFrame: DataFrame con todas las películas y sus características
        """
        print("📥 Cargando datos...")
        
        try:
            movies = pd.read_csv(os.path.join(self.data_dir, 'movies_metadata.csv'))
            keywords = pd.read_csv(os.path.join(self.data_dir, 'keywords.csv'))
            credits = pd.read_csv(os.path.join(self.data_dir, 'credits.csv'), low_memory=False)
            
            # Fusiona los dataframes
            movies = movies.merge(credits, on='id', how='left')
            movies = movies.merge(keywords, on='id', how='left')
            
            # Limpia valores nulos
            movies['cast'] = movies['cast'].fillna('[]')
            movies['crew'] = movies['crew'].fillna('[]')
            movies['keywords'] = movies['keywords'].fillna('[]')
            movies['genres'] = movies['genres'].fillna('[]')
            
            # Parsea las columnas JSON
            movies['cast'] = movies['cast'].apply(self._parse_json)
            movies['crew'] = movies['crew'].apply(self._parse_json)
            movies['keywords'] = movies['keywords'].apply(self._parse_json)
            movies['genres'] = movies['genres'].apply(self._parse_json)
            
            # Elimina películas sin título
            movies = movies[movies['title'].notna()].reset_index(drop=True)
            
            print(f"✅ Se cargaron {len(movies)} películas")
            return movies
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("Asegúrate de que los archivos CSV están en la carpeta 'data/'")
            return None
    
    @staticmethod
    def _parse_json(x):
        """Parsea strings JSON de forma segura"""
        try:
            return ast.literal_eval(x) if isinstance(x, str) else x
        except (ValueError, SyntaxError):
            return []
    
    def extract_features(self):
        """
        Extrae características clave de las películas
        
        Returns:
            list: Lista con los nombres de los directores
        """
        print("🔧 Extrayendo características...")
        
        # Extrae actores principales
        self.movies['actors'] = self.movies['cast'].apply(
            lambda x: ' '.join([actor['name'] for actor in x[:3]]) if x else ''
        )
        
        # Extrae géneros
        self.movies['genres_str'] = self.movies['genres'].apply(
            lambda x: ' '.join([genre['name'] for genre in x]) if x else ''
        )
        
        # Extrae palabras clave
        self.movies['keywords_str'] = self.movies['keywords'].apply(
            lambda x: ' '.join([kw['name'] for kw in x]) if x else ''
        )
        
        # Extrae directores
        directors = []
        for crew in self.movies['crew']:
            director = [c['name'] for c in crew if c['job'] == 'Director']
            directors.append(' '.join(director) if director else '')
        self.movies['director'] = directors
        
        return directors
    
    def create_metadata_soup(self):
        """
        Crea una 'sopa de metadatos' combinando todas las características
        """
        print("🍲 Creando matriz de características...")
        
        # Combina todas las características
        self.movies['soup'] = (
            self.movies['director'] + ' ' +
            self.movies['actors'] + ' ' +
            self.movies['genres_str'] + ' ' +
            self.movies['keywords_str']
        )
        
        # Convierte a minúsculas y crea la matriz
        count = CountVectorizer(stop_words='english', max_features=5000)
        count_matrix = count.fit_transform(self.movies['soup'])
        
        # Calcula similitud del coseno
        print("📊 Calculando matriz de similitud...")
        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)
        
        print(f"✅ Matriz de similitud completada: {self.cosine_sim.shape}")
    
    def calculate_weighted_rating(self):
        """
        Calcula una puntuación ponderada basada en votos y rating
        """
        print("⭐ Calculando puntuaciones ponderadas...")
        
        # Convierte a numéricas y rellena nulos
        self.movies['vote_count'] = pd.to_numeric(self.movies['vote_count'], errors='coerce').fillna(0)
        self.movies['vote_average'] = pd.to_numeric(self.movies['vote_average'], errors='coerce').fillna(0)
        
        C = self.movies['vote_average'].mean()
        m = self.movies['vote_count'].quantile(0.95)
        
        def weighted_rating(row):
            v = row['vote_count']
            R = row['vote_average']
            return (v / (v + m) * R) + (m / (v + m) * C) if (v + m) > 0 else 0
        
        self.movies['weighted_rating'] = self.movies.apply(weighted_rating, axis=1)
    
    def load_and_prepare_data(self):
        """Carga los datos y prepara el modelo"""
        self.movies = self.load_and_clean_data()
        
        if self.movies is not None:
            self.extract_features()
            self.create_metadata_soup()
            self.calculate_weighted_rating()
            print("\n🎬 Sistema de recomendación listo!")
    
    def get_recommendations(self, title, n_recommendations=10):
        """
        Obtiene recomendaciones para una película
        
        Args:
            title (str): Título de la película
            n_recommendations (int): Número de recomendaciones
            
        Returns:
            pd.DataFrame: DataFrame con las películas recomendadas
        """
        if self.movies is None or self.cosine_sim is None:
            print("❌ El modelo no está inicializado correctamente")
            return None
        
        # Busca el índice de la película
        matches = self.movies[self.movies['title'].str.lower() == title.lower()]
        
        if matches.empty:
            print(f"❌ No se encontró la película '{title}'")
            print("\n💡 Películas disponibles similares:")
            similar_titles = self.movies[self.movies['title'].str.contains(title, case=False, na=False)]
            return similar_titles[['title', 'weighted_rating']].head(10)
        
        idx = matches.index[0]
        
        # Obtiene las películas más similares
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations + 50]  # Más películas para filtrar
        
        # Índices de las películas similares
        movie_indices = [i[0] for i in sim_scores]
        
        # Filtra por puntuación ponderada
        recommended_movies = self.movies.iloc[movie_indices].sort_values(
            'weighted_rating', ascending=False
        )
        
        return recommended_movies[[
            'title', 'release_date', 'vote_average', 'weighted_rating'
        ]].head(n_recommendations)
    
    def get_popular_movies(self, n_movies=10, min_votes=100):
        """
        Obtiene las películas más populares
        
        Args:
            n_movies (int): Número de películas
            min_votes (int): Mínimo número de votos
            
        Returns:
            pd.DataFrame: DataFrame con películas populares
        """
        popular = self.movies[self.movies['vote_count'] >= min_votes].sort_values(
            'weighted_rating', ascending=False
        )
        
        return popular[['title', 'release_date', 'vote_average', 'weighted_rating']].head(n_movies)
    
    def get_recommendations_by_genre(self, genre, n_movies=10):
        """
        Obtiene películas recomendadas por género
        
        Args:
            genre (str): Género de películas
            n_movies (int): Número de películas
            
        Returns:
            pd.DataFrame: DataFrame con películas del género
        """
        genre_movies = self.movies[
            self.movies['genres_str'].str.contains(genre, case=False, na=False)
        ].sort_values('weighted_rating', ascending=False)
        
        return genre_movies[['title', 'release_date', 'vote_average', 'weighted_rating']].head(n_movies)


def main():
    """Función principal para interactuar con el sistema"""
    
    print("=" * 60)
    print("🎬 SISTEMA DE RECOMENDACIÓN DE PELÍCULAS 🎬")
    print("=" * 60 + "\n")
    
    # Inicializa el recomendador
    recommender = MovieRecommender()
    
    if recommender.movies is None:
        return
    
    while True:
        print("\n📌 OPCIONES:")
        print("1. Obtener recomendaciones basadas en una película")
        print("2. Ver películas populares")
        print("3. Buscar películas por género")
        print("4. Salir")
        
        choice = input("\n👉 Selecciona una opción (1-4): ").strip()
        
        if choice == '1':
            title = input("Ingresa el título de la película: ").strip()
            n = int(input("¿Cuántas recomendaciones? (default 10): ") or 10)
            
            print(f"\n🔍 Buscando recomendaciones para '{title}'...\n")
            recommendations = recommender.get_recommendations(title, n)
            
            if not recommendations.empty:
                print(recommendations.to_string(index=False))
        
        elif choice == '2':
            n = int(input("¿Cuántas películas? (default 10): ") or 10)
            print("\n🌟 PELÍCULAS MÁS POPULARES\n")
            popular = recommender.get_popular_movies(n)
            print(popular.to_string(index=False))
        
        elif choice == '3':
            genre = input("Ingresa el género: ").strip()
            n = int(input("¿Cuántas películas? (default 10): ") or 10)
            
            print(f"\n🎭 PELÍCULAS DE {genre.upper()}\n")
            genre_movies = recommender.get_recommendations_by_genre(genre, n)
            print(genre_movies.to_string(index=False))
        
        elif choice == '4':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("❌ Opción no válida. Intenta de nuevo.")


if __name__ == "__main__":
    main()
