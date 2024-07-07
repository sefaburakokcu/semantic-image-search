# Semantic Image Search

## Overview
`semantic-image-search` is a web-based application that allows users to perform semantic searches on a collection of images using either textual descriptions or image inputs. 
It leverages the OpenAI's CLIP model for embedding extraction and Milvus for efficient similarity search.

![Demo SIS]("https://github.com/sefaburakokcu/semantic-image-search/tree/main/assets/demo.gif")


## Features
- **Collection Management:** Create, upload, and delete image collections.
- **Search Capabilities:** Perform image and text-based searches on image collections.
- **Efficient Storage and Retrieval:** Utilize Milvus for fast and scalable vector similarity search.

## Installation

### Prerequisites
- Python 3.8+
- pytorch
- streamlit
- transformers
- pymilvus

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/sefaburakokcu/semantic-image-search.git
   cd semantic-image-search
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the application:
   ```sh
   streamlit run app.py  --server.maxUploadSize 2000
   ```

## Usage

### Sidebar: Collection Management
- **Create Collection:** Enter a name for the new collection and click "Create Collection."
- **Upload Images to Collection:** Select an existing collection, upload images or a zip file containing images, and click "Upload Images."
- **Delete Collection:** Select a collection to delete and click "Delete Collection."

### Main Interface: Image and Text Search
- **Search Configuration:**
  - Select the collection to search.
  - Set the number of results and the number of results to display per row.
  - Choose between text search and image search.
- **Search Input:**
  - For text search, enter a search query.
  - For image search, upload an image file.
- **Search Results:** Click "Search" to view results. Images are displayed with their similarity percentage.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit for the web application framework](https://streamlit.io/)
- [Milvus for the vector database](https://github.com/milvus-io/milvus-lite)
- [OpenAI for the CLIP model from HuggingFace transformers](https://huggingface.co/docs/transformers/index)

For any questions or issues, please open an issue on GitHub.