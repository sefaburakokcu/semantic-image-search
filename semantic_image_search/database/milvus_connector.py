import logging
from pymilvus import MilvusClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusConnector:
    def __init__(self, db_name="products.db"):
        self.client = MilvusClient(db_name)

    def create_collection(self, collection_name, dimension=512, recreate=False):
        if self.client.has_collection(collection_name):
            logger.info(f"Collection {collection_name} already exists.")
            if not recreate:
                return
            self.delete_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vector_field_name="vector",
            dimension=dimension,
            auto_id=True,
            enable_dynamic_field=True,
            metric_type="COSINE",
        )
        logger.info(f"Collection {collection_name} created.")

    def delete_collection(self, collection_name):
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
            logger.info(f"Collection {collection_name} deleted.")
        else:
            logger.info(f"Collection {collection_name} does not exist.")

    def list_collections(self):
        collections = self.client.list_collections()
        logger.info("Collections: %s", collections)
        return collections

    def add_data(self, collection_name, data):
        if not self.client.has_collection(collection_name):
            logger.info(f"Collection {collection_name} does not exist.")
            return
        self.client.insert(collection_name=collection_name, data=data)
        logger.info(f"Added {data} to collection {collection_name}.")

    def search_data(self, collection_name, vector, output_fields, top_k=10):
        if not self.client.has_collection(collection_name):
            logger.info(f"Collection {collection_name} does not exist.")
            return
        results = self.client.search(
            collection_name=collection_name,
            data=vector,
            limit=top_k,
            output_fields=output_fields
        )
        return results


if __name__ == "__main__":
    import numpy as np

    # Initialize the MilvusConnector with the database name
    connector = MilvusConnector(db_name="products.db")

    # Create a new collection with a specified dimension
    connector.create_collection("example_collection", dimension=128)

    # Add data to the collection
    image_data = {
        "vector": np.random.rand(128).tolist(),  # Example vector with 128 dimensions
        "image_name": "image1.jpg",
        "image_path": "/path/to/image1.jpg"
    }
    connector.add_data("example_collection", image_data)

    # List all collections
    connector.list_collections()

    # Search for a vector in the collection
    search_vector = np.random.rand(128).tolist()
    connector.search_data("example_collection", [search_vector], ["image_name", "image_path"])

    # Delete the collection
    connector.delete_collection("example_collection")
