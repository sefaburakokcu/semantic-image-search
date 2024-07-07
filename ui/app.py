import sys
import os
import glob
import zipfile
import shutil
import streamlit as st

from PIL import Image

sys.path.append("..")
from semantic_image_search.database.milvus_connector import MilvusConnector
from semantic_image_search.models.embedder import EmbeddingExtractor

st.set_page_config(page_title="Semantic Image Search", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

DATA_SAVE_FOLDER = "../data"
DB_NAME = "../data/search.db"
IMG_WIDTH = 300


@st.cache_data
def initialize_model():
    embedding_extractor = EmbeddingExtractor("openai/clip-vit-base-patch32", "clip")
    return embedding_extractor

@st.cache_resource
def initialize_db():
    db_connector = MilvusConnector(DB_NAME)
    return db_connector


def save_image(image, collection_name, image_name):
    collection_path = os.path.join(DATA_SAVE_FOLDER, collection_name)
    os.makedirs(collection_path, exist_ok=True)
    image_path = os.path.join(collection_path, image_name)
    image.save(image_path)
    return image_path


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


embedding_extractor = initialize_model()
db_connector = initialize_db()

#----------------------------------Sidebar:Collection Management-----------------------------------#
st.sidebar.subheader("Collection Management")

menu_option = st.sidebar.selectbox("Menu", ["Create Collection", "Upload Images to Collection", "Delete Collection"])

if menu_option == "Create Collection":
    # Add new collection
    new_collection_name = st.sidebar.text_input("New Collection Name")
    if st.sidebar.button("Create Collection"):
        if new_collection_name:
            try:
                db_connector.create_collection(new_collection_name, dimension=512)
                st.sidebar.success(f"Collection '{new_collection_name}' created successfully.")
            except Exception as e:
                st.error(f'Cannot create {new_collection_name} collection.')
        else:
            st.sidebar.error("Please enter a collection name.")

elif menu_option == "Upload Images to Collection":
    with st.sidebar:
        with st.form("upload-images-form", clear_on_submit=True):
            collections = db_connector.list_collections()
            selected_collection = st.selectbox("Select Collection", collections)

            # Upload images or zip
            uploaded_files = st.file_uploader("Upload Images or Zip", type=["png", "jpg", "jpeg", "zip"],
                                              accept_multiple_files=True)
            upload_images = st.form_submit_button("Upload Images")

        if upload_images:
            if uploaded_files:
                progress_bar = st.progress(0)
                total_files = len(uploaded_files)
                file_count = 0

                for uploaded_file in uploaded_files:
                    if uploaded_file.name.endswith('.zip'):
                        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                            zip_ref.extractall("temp_upload")
                            extracted_files = glob.glob("temp_upload/**/*", recursive=True)
                            total_files += len(extracted_files)
                            for file_path in extracted_files:
                                if os.path.isfile(file_path):
                                    image_name = os.path.basename(file_path)
                                    image = Image.open(file_path)
                                    image_path = save_image(image, selected_collection, image_name)
                                    image_embeddings = embedding_extractor.extract_image_embeddings([image])[0]
                                    data = {
                                        "vector": image_embeddings,
                                        "image_name": image_name,
                                        "image_path": image_path
                                    }
                                    db_connector.add_data(selected_collection, data)
                                # Update the progress bar
                                file_count += 1
                                progress_bar.progress(file_count / total_files)
                    else:
                        image_name = uploaded_file.name
                        image = Image.open(uploaded_file)
                        image_path = save_image(image, selected_collection, image_name)
                        image_embeddings = embedding_extractor.extract_image_embeddings([image])[0]
                        data = {
                            "vector": image_embeddings,
                            "image_name": image_name,
                            "image_path": image_path
                        }
                        db_connector.add_data(selected_collection, data)

                    # Update the progress bar
                    file_count += 1
                    progress_bar.progress(file_count / total_files)

                st.sidebar.success("Images uploaded successfully.")
            else:
                st.warning("No image selected.")

elif menu_option == "Delete Collection":
    # Collection selection for deleting
    collections = db_connector.list_collections()
    delete_collection_name = st.sidebar.selectbox("Delete Collection", collections)
    if st.sidebar.button("Delete Collection"):
        if delete_collection_name:
            db_connector.delete_collection(delete_collection_name)
            collection_path = os.path.join("data", delete_collection_name)
            if os.path.exists(collection_path):
                shutil.rmtree(collection_path)
            st.sidebar.success(f"Collection '{delete_collection_name}' deleted successfully.")
        else:
            st.sidebar.error("Please select a collection to delete.")


#----------------------------------Main: Image and Text Search"-----------------------------------#
st.title("Semantic Image Search")
st.sidebar.divider()
# Sidebar elements
st.sidebar.header("Search Configuration")
collections = db_connector.list_collections()
selected_collection = st.sidebar.selectbox("Select Collection for Search", collections)
top_n = st.sidebar.number_input("Number of Results", min_value=1, max_value=100, value=10)
num_cols = st.sidebar.number_input("Number of Results in a Row", min_value=1, max_value=10, value=5)

search_type = st.sidebar.radio("Select Search Type", ("Text Search", "Image Search"))

# Main area elements
if search_type == "Text Search":
    search_text = st.text_input("Enter Text for Search")
elif search_type == "Image Search":
    search_image = st.file_uploader("Upload Image for Search", type=["png", "jpg", "jpeg"])

if st.button("Search"):
    if search_type == "Image Search" and search_image:
        image = Image.open(search_image)
        search_embeddings = embedding_extractor.extract_image_embeddings([image])
    elif search_type == "Text Search" and search_text:
        search_embeddings = embedding_extractor.extract_text_embeddings([search_text])
    else:
        st.warning("Please provide the required input for the selected search type!")
        search_embeddings = None

    if search_embeddings is not None:
        image_list = []
        caption_list = []
        search_results = db_connector.search_data(selected_collection, search_embeddings, ["image_name", "image_path"], top_n)
        for result in search_results[0]:
            image_list.append(result["entity"]["image_path"])
            caption_list.append(f"{int(result['distance']*100)}%")
            # st.image(result["entity"]["image_path"], caption=result["entity"]["image_name"])

        for i, (image_chunk, caption_chunks) in enumerate(zip(chunks(image_list, num_cols), chunks(caption_list, num_cols))):
            cols = st.columns(num_cols, gap="small")
            for idx, col in enumerate(cols):
                if idx < len(image_chunk):
                    col.image(image_chunk[idx], caption=caption_chunks[idx], width=IMG_WIDTH)
