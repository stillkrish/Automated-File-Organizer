import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import os
import shutil
import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class ClusterMaker:
    """
    A class to cluster document embeddings and organize files into folders.
    """
    
    def __init__(self, min_clusters=2, max_clusters=10):
        """
        Initialize the ClusterMaker.
        
        Args:
            min_clusters (int): Minimum number of clusters to try
            max_clusters (int): Maximum number of clusters to try
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.labels = None
        self.model = None
        self.pca = None
        
    def find_optimal_clusters(self, embeddings):
        """
        Find the optimal number of clusters using silhouette score.
        
        Args:
            embeddings (np.array): Document embeddings
            
        Returns:
            int: Optimal number of clusters
        """
        if len(embeddings) < self.min_clusters:
            return 1
            
        scores = []
        for k in range(self.min_clusters, min(self.max_clusters + 1, len(embeddings))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # Skip silhouette calculation if there's only one label
            if len(np.unique(labels)) < 2:
                scores.append(-1)
                continue
                
            score = silhouette_score(embeddings, labels)
            scores.append(score)
            print(f"Clusters: {k}, Silhouette Score: {score:.4f}")
            
        if not scores:
            return 1
            
        best_k = range(self.min_clusters, min(self.max_clusters + 1, len(embeddings)))[np.argmax(scores)]
        return best_k
        
    def cluster_embeddings(self, embeddings, method="kmeans", optimal=True):
        """
        Cluster document embeddings.
        
        Args:
            embeddings (np.array): Document embeddings
            method (str): Clustering method, either "kmeans" or "dbscan"
            optimal (bool): Whether to find optimal number of clusters
            
        Returns:
            np.array: Cluster labels for each document
        """
        if len(embeddings) <= 1:
            self.labels = np.zeros(len(embeddings), dtype=int)
            return self.labels
            
        if method == "kmeans":
            if optimal:
                n_clusters = self.find_optimal_clusters(embeddings)
            else:
                n_clusters = min(self.max_clusters, len(embeddings))
                
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.labels = self.model.fit_predict(embeddings)
            
        elif method == "dbscan":
            self.model = DBSCAN(eps=0.5, min_samples=5)
            self.labels = self.model.fit_predict(embeddings)
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
            
        return self.labels
        
    def visualize_clusters(self, embeddings, file_paths, output_path="cluster_visualization.png"):
        if self.labels is None:
            raise ValueError("Run cluster_embeddings first")
            
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        if embeddings.shape[1] > 2:
            self.pca = PCA(n_components=2)
            reduced_embeddings = self.pca.fit_transform(embeddings)
        else:
            reduced_embeddings = embeddings
            
        plt.figure(figsize=(12, 8))
        
        # Get unique labels and assign colors
        unique_labels = set(self.labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = self.labels == label
            plt.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                c=color.reshape(1, -1),
                label=f"Cluster {label}",
                alpha=0.7
            )
            
        plt.legend()
        plt.title("Document Clusters")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.savefig(output_path)
        plt.close()
        
    def get_cluster_keywords(self, texts, top_n=5):
        """
        Get top keywords for each cluster.
        
        Args:
            texts (list): List of document texts
            top_n (int): Number of top keywords to return
            
        Returns:
            dict: Cluster labels mapped to their top keywords
        """
        if self.labels is None:
            raise ValueError("Run cluster_embeddings first")
            
        # Extract keywords using TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        keywords = {}
        for label in set(self.labels):
            # Get indices of documents in this cluster
            cluster_indices = np.where(self.labels == label)[0]
            
            # Skip empty clusters
            if len(cluster_indices) == 0:
                keywords[label] = ["unknown"]
                continue
                
            # Get TF-IDF scores for this cluster
            cluster_tfidf = tfidf_matrix[cluster_indices].toarray().mean(axis=0)
            
            # Get top keywords
            top_indices = cluster_tfidf.argsort()[-top_n:][::-1]
            top_keywords = [feature_names[i] for i in top_indices]
            
            keywords[label] = top_keywords
            
        return keywords
        
    def create_folders(self, output_dir, file_paths, move_files=False):
        """
        Create folders based on clusters and organize files.
        
        Args:
            output_dir (str): Output directory
            file_paths (list): List of file paths
            move_files (bool): Whether to move files instead of copying
            
        Returns:
            dict: Mapping of cluster labels to folder paths
        """
        if self.labels is None:
            raise ValueError("Run cluster_embeddings first")
            
        # Extract texts from file paths to generate keywords
        from TextProcessor import TextProcessor
        processor = TextProcessor("")
        texts = []
        for path in file_paths:
            text = processor.extract_text_from_file(path)
            clean_text = processor.clean_text(text)
            texts.append(clean_text)
            
        # Get cluster keywords
        keywords = self.get_cluster_keywords(texts)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create cluster folders
        folder_paths = {}
        for label, kws in keywords.items():
            folder_name = f"Cluster_{label}_{'-'.join(kws[:3])}"
            # Replace invalid characters for folder names
            folder_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in folder_name)
            folder_path = os.path.join(output_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            folder_paths[label] = folder_path
            
        # Copy/move files to appropriate folders
        file_organization = []
        for i, (path, label) in enumerate(zip(file_paths, self.labels)):
            dest_folder = folder_paths[label]
            filename = os.path.basename(path)
            dest_path = os.path.join(dest_folder, filename)
            
            # Handle duplicate filenames
            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(filename)
                dest_path = os.path.join(dest_folder, f"{name}_{counter}{ext}")
                counter += 1
                
            # Copy or move the file
            try:
                if move_files:
                    shutil.move(path, dest_path)
                else:
                    shutil.copy2(path, dest_path)
                
                file_organization.append({
                    'original_path': path,
                    'new_path': dest_path,
                    'cluster': label,
                    'cluster_keywords': keywords[label]
                })
            except Exception as e:
                print(f"Error organizing file {path}: {str(e)}")
                
        # Create organization CSV
        df = pd.DataFrame(file_organization)
        csv_path = os.path.join(output_dir, "file_organization.csv")
        df.to_csv(csv_path, index=False)
        
        return folder_paths
        
    def summarize_clusters(self, file_paths):
        """
        Create a summary of clusters and their contents.
        
        Args:
            file_paths (list): List of file paths
            
        Returns:
            dict: Cluster summary
        """
        if self.labels is None:
            raise ValueError("Run cluster_embeddings first")
            
        summary = {}
        for label in set(self.labels):
            cluster_indices = np.where(self.labels == label)[0]
            cluster_files = [file_paths[i] for i in cluster_indices]
            
            # Get file extensions
            extensions = [os.path.splitext(path)[1].lower() for path in cluster_files]
            extension_counts = Counter(extensions)
            
            summary[label] = {
                'file_count': len(cluster_files),
                'file_extensions': dict(extension_counts),
                'files': [os.path.basename(path) for path in cluster_files]
            }
            
        return summary


# Example usage
if __name__ == "__main__":
    # This is just a demonstration
    from TextProcessor import TextProcessor
    from TextEmbedder import TextEmbedder
    import numpy as np
    
    # Create random embeddings for testing
    embeddings = np.random.rand(10, 128)
    file_paths = [f"file_{i}.txt" for i in range(10)]
    
    # Test clustering
    cluster_maker = ClusterMaker()
    labels = cluster_maker.cluster_embeddings(embeddings)
    
    print(f"Cluster labels: {labels}")
    print(f"Number of clusters: {len(set(labels))}")