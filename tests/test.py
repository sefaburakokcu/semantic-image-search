import sys
import os
import glob

sys.path.append("..")
from PIL import Image
from semantic_image_search.database.milvus_connector import MilvusConnector
from semantic_image_search.models.embedder import EmbeddingExtractor

embedding_extractor = EmbeddingExtractor("openai/clip-vit-base-patch32", "clip")
db_connector = MilvusConnector()
products_list_name = "kaggle_products"
db_connector.create_collection(products_list_name, dimension=512, recreate=True)

INPUT_IMAGES = "../inputs/"

image_paths = glob.glob(INPUT_IMAGES+"*")

for image_path in image_paths:
    image_name = os.path.basename(image_path)
    image = Image.open(image_path)
    image_embeddings = embedding_extractor.extract_image_embeddings([image])[0]
    data = {
        "vector": image_embeddings,
        "image_name": image_name,
        "image_path": image_path
    }
    db_connector.add_data(products_list_name, data)

db_connector.search_data(products_list_name, [image_embeddings], ["image_name", "image_path"])

db_connector.delete_collection(products_list_name)
