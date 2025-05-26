import os
import json
import faiss
import numpy as np
import logging
from feature_extractor import ImageFeatureExtractor

logger = logging.getLogger(__name__)

class ImageRetrievalSystem:
    """
    Builds a FAISS Inner-Product index over object-cropped, L2-normalized ViT features.
    """
    def __init__(self, index_path=None, metadata_path=None, use_gpu=False, heavy_model=True, device=None):
        # Initialize feature extractor
        self.feature_extractor = ImageFeatureExtractor(heavy_model=heavy_model, device=device)
        self.feature_dim = self.feature_extractor.feature_dim

        # Metadata mapping from index -> image path
        self.metadata = {}

        # Create an Inner-Product FAISS index (cosine similarity)
        self.index = faiss.IndexFlatIP(self.feature_dim)
        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        # Load existing index if provided
        if index_path and metadata_path:
            self.load(index_path, metadata_path)

    def index_images(self, directory):
        """
        Extract features for each .jpg in `directory` and add to FAISS index.
        """
        paths = [os.path.join(directory, f)
                 for f in os.listdir(directory)
                 if f.lower().endswith('.jpg')]
        features = []
        for path in paths:
            try:
                feat = self.feature_extractor.extract_features(path)
                features.append(feat)
                idx = len(self.metadata)
                self.metadata[str(idx)] = {'path': path}
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")

        if not features:
            raise ValueError("No features extracted to index.")

        # Stack and normalize (precaution) then add to FAISS
        features_np = np.stack(features)
        faiss.normalize_L2(features_np)
        self.index.add(features_np)

    def search(self, query_image, k=5):
        """
        Returns top-k results sorted by cosine similarity descending.
        Each entry is (path, similarity, distance).
        """
        # 1) Extract & normalize query feature
        feat = self.feature_extractor.extract_features(query_image).reshape(1, -1)
        faiss.normalize_L2(feat)

        # 2) Search FAISS (Inner-Product = cosine similarity for unit vectors)
        sims, idxs = self.index.search(feat, k)

        # 3) Build list of (path, sim)
        results = [
            (self.metadata[str(i)]['path'], float(sim))
            for i, sim in zip(idxs[0], sims[0])
        ]

        # 4) Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        # 5) Compute distance = 1 - sim, and return full tuples
        return [(path, sim, 1.0 - sim) for path, sim in results]

    def save(self, index_path, metadata_path):
        """Save FAISS index and metadata JSON to disk."""
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)

    def load(self, index_path, metadata_path):
        """Load FAISS index and metadata JSON from disk."""
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
