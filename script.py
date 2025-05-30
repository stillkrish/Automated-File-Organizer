import os
import numpy as np
import pandas as pd
from TextProcessor import TextProcessor
from TextEmbedder import TextEmbedder
from ClusterMaker import ClusterMaker
import argparse
import time

"""Take a list of file paths and prints each one out"""
def print_files(file_paths):
    for f in file_paths:
        print(f)

def organize_files(input_dir, output_dir, min_clusters=2, max_clusters=10, move_files=False, visualize=True):
    """
    Organize files using text embeddings and clustering.
    
    Args:
        input_dir (str): Input directory containing files to organize
        output_dir (str): Output directory for organized files
        min_clusters (int): Minimum number of clusters
        max_clusters (int): Maximum number of clusters
        move_files (bool): Whether to move files instead of copying
        visualize (bool): Whether to generate visualization
    """
    print(f"Starting file organization from {input_dir}")
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize processors
    text_processor = TextProcessor(input_dir)
    text_embedder = TextEmbedder()
    
    # Get all file paths
    file_paths = text_processor.get_all_file_paths()
    print(f"Found {len(file_paths)} files")
    
    # Process files
    embeddings = []
    valid_files = []
    texts = []
    
    print("Processing files...")
    for i, file_path in enumerate(file_paths):
        if i % 10 == 0:
            print(f"Processing file {i+1}/{len(file_paths)}")
            
        try:
            # Extract and clean text
            text = text_processor.extract_text_from_file(file_path)
            if not text:
                continue
                
            clean_text = text_processor.clean_text(text)
            if not clean_text:
                continue
                
            # Create chunks and embeddings
            results = text_embedder.process_text(clean_text)
            if not results:
                continue
                
            # Get the embeddings from the results
            _, chunk_embeddings = zip(*results)
            
            # Average embeddings across chunks
            avg_embedding = text_embedder.average_embeddings(chunk_embeddings)
            
            # Add to our lists
            embeddings.append(avg_embedding)
            valid_files.append(file_path)
            texts.append(clean_text)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Convert embeddings to numpy array
    if not embeddings:
        print("No valid embeddings found. Exiting.")
        return
        
    embeddings_array = np.array(embeddings)
    print(f"Generated {len(embeddings_array)} embeddings")
    
    # Cluster embeddings
    print("Clustering files...")
    cluster_maker = ClusterMaker(min_clusters=min_clusters, max_clusters=max_clusters)
    labels = cluster_maker.cluster_embeddings(embeddings_array)
    
    # Get cluster summary
    summary = cluster_maker.summarize_clusters(valid_files)
    print("\nCluster Summary:")
    for label, info in summary.items():
        print(f"Cluster {label}: {info['file_count']} files")
        print(f"Extensions: {info['file_extensions']}")
        
    # Create visualization
    if visualize:
        print("Generating visualization...")
        cluster_maker.visualize_clusters(embeddings_array, valid_files, 
                                         output_path=os.path.join(output_dir, "cluster_visualization.png"))
    
    # Create folders and organize files
    print(f"Organizing files into {output_dir}...")
    folder_paths = cluster_maker.create_folders(output_dir, valid_files, move_files)
    
    end_time = time.time()
    print(f"File organization completed in {end_time - start_time:.2f} seconds")
    print(f"Results saved to {output_dir}/file_organization.csv")

# Run the script
if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="AI File Organizer")
    parser.add_argument("--input", help="Input directory containing files to organize")
    parser.add_argument("--output", help="Output directory for organized files", default="./organized_files")
    parser.add_argument("--min-clusters", type=int, default=2, help="Minimum number of clusters")
    parser.add_argument("--max-clusters", type=int, default=10, help="Maximum number of clusters")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    
    args = parser.parse_args()
    
    # If input not provided, ask for it
    input_dir = args.input
    if not input_dir:
        input_dir = input("Please enter the directory name: ")
        while not os.path.isdir(input_dir):
            print("Invalid Directory Name")
            input_dir = input("Please enter the directory name: ")
    
    # Run file organization
    organize_files(
        input_dir=input_dir,
        output_dir=args.output,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        move_files=args.move,
        visualize=not args.no_viz
    )