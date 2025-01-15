from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
import os

# Configuration
data_folder = './data'
collection_name = 'tata_car_manuals'
qdrant_url = "http://localhost:6333"  

# Initialize Qdrant client
client = QdrantClient(url=qdrant_url)

# Initialize HuggingFace BGE embeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def create_and_upload_embeddings():
    # Check if the collection exists, create if not
    collections = client.get_collections().collections
    if collection_name not in [col.name for col in collections]:
        vector_params = VectorParams(
            size=1024,  # Embedding size
            distance="Cosine"  # Similarity metric
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_params
        )

    global_idx = 0  # Global counter for unique IDs across all files

    # Process files in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):
            car_model = filename.split('.')[0]
            try:
                with open(os.path.join(data_folder, filename), 'r', encoding='utf-8', errors='ignore') as file:
                    chunks = file.readlines()  # Read all lines as chunks
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            # Process each chunk
            for chunk in chunks:
                chunk = chunk.strip()  # Remove leading/trailing whitespace
                if not chunk:  # Skip empty chunks
                    continue
                
                # Generate embedding
                embedding = embeddings.embed_documents([chunk])[0]

                # Create metadata
                metadata = {
                    'car_model': car_model,
                    'chunk_index': global_idx,  # Unique global index
                    'source_file': filename,
                    'chunk_text': chunk
                }

                # Upload to Qdrant
                client.upsert(
                    collection_name=collection_name,
                    points=[PointStruct(id=global_idx, vector=embedding, payload=metadata)]
                )
                print(f"Uploaded chunk {global_idx + 1} of {car_model} to Qdrant with metadata and chunk text")

                global_idx += 1  # Increment global counter

if __name__ == "__main__":
    create_and_upload_embeddings()











'''



from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
import os
import uuid

data_folder = './data'
collection_name = 'tata_car_manuals'
qdrant_url = "http://localhost:6333"

# Initialize Qdrant client
client = QdrantClient(url=qdrant_url)

# Initialize HuggingFace BGE embeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def create_and_upload_embeddings():
    # Check if the collection exists and create if not
    collections = client.get_collections().collections
    if collection_name not in [col.name for col in collections]:
        vector_params = VectorParams(
            size=1024,  # Dimensionality of the embedding vectors
            distance="Cosine"  # Distance metric for vector similarity
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_params
        )

    # Process files in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):
            car_model = filename.split('.')[0]
            try:
                with open(os.path.join(data_folder, filename), 'r', encoding='utf-8', errors='ignore') as file:
                    chunks = file.readlines()  # Read the chunks of the file
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            print(f"Processing {len(chunks)} chunks for {car_model}...")  # Log chunk count for each car model

            # Iterate through each chunk and upload to Qdrant
            for idx, chunk in enumerate(chunks):
                chunk = chunk.strip()  # Remove leading/trailing whitespace
                if not chunk:  # Skip empty chunks
                    continue
                
                embedding = embeddings.embed_documents([chunk])[0]  # Get the embedding for the chunk

                # Generate a UUID for each chunk
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{car_model}_{idx}"))
                print(f"Generated Point ID: {point_id} for chunk {idx + 1} of {car_model}")  # Log the point ID
                
                # Create metadata for the chunk
                metadata = {
                    'car_model': car_model,
                    'chunk_text': chunk  # Store the chunk text in metadata
                }

                # Upload the new point to Qdrant
                try:
                    client.upsert(
                        collection_name=collection_name,
                        points=[PointStruct(id=point_id, vector=embedding, payload=metadata)]
                    )
                    print(f"Uploaded chunk {idx + 1} of {car_model} to Qdrant with metadata and chunk text")
                except Exception as e:
                    print(f"Error uploading chunk {idx + 1} of {car_model}: {e}")

if __name__ == "__main__":
    create_and_upload_embeddings()

'''



'''


from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
import os

data_folder = './data'
collection_name = 'tata_car_manuals'
qdrant_url = "http://localhost:6333"  

# Initialize Qdrant client
client = QdrantClient(url=qdrant_url)

# Initialize HuggingFace BGE embeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def create_and_upload_embeddings():
    collections = client.get_collections().collections
    if collection_name not in [col.name for col in collections]:
        vector_params = VectorParams(
            size=1024,  
            distance="Cosine"  
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_params
        )


    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):
            car_model = filename.split('.')[0]
            try:
                with open(os.path.join(data_folder, filename), 'r', encoding='utf-8', errors='ignore') as file:
                    chunks = file.readlines()  
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

        
            for idx, chunk in enumerate(chunks):
                chunk = chunk.strip()  
                if not chunk:  
                    continue
                
              
                embedding = embeddings.embed_documents([chunk])[0]

               
                metadata = {
                    'car_model': car_model,
                    'chunk_index': idx,
                    'source_file': filename,
                    'chunk_text': chunk  
                }

                client.upsert(
                    collection_name=collection_name,
                    points=[PointStruct(id=idx, vector=embedding, payload=metadata)]
                )
                print(f"Uploaded chunk {idx + 1} of {car_model} to Qdrant with metadata and chunk text")

if __name__ == "__main__":
    create_and_upload_embeddings()


'''
