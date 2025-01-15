import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import subprocess

# Initialize Qdrant client
collection_name = 'tata_car_manuals'
qdrant_url = "http://localhost:6333"
client = QdrantClient(url=qdrant_url)

# Set up embeddings model
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Function to perform similarity search
def similarity_search(query: str, car_model: str, top_k: int = 3, score_threshold: float = 0.8):
    query_embedding = embeddings.embed_documents([query])[0]

    car_model_filter = Filter(
        must=[FieldCondition(key="car_model", match=MatchValue(value=car_model))]
    )

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        query_filter=car_model_filter
    )

    results = []
    for result in search_result:
        chunk_text = result.payload.get('chunk_text', 'No text available')
        results.append({
            'score': result.score,
            'car_model': result.payload.get('car_model', 'Unknown'),
            'chunk_index': result.payload.get('chunk_index', -1),
            'source_file': result.payload.get('source_file', 'Unknown'),
            'chunk_text': chunk_text
        })

    high_score_results = [r for r in results if r['score'] >= score_threshold]

    if not high_score_results and results:
        high_score_results = [max(results, key=lambda x: x['score'])]

    return high_score_results

# Function to query the LLM with Ollama
def query_to_llm_with_ollama(query, results):
    context = "\n".join([f"Chunk {i+1} (Car Model: {result['car_model']}): {result['chunk_text']}" 
                         for i, result in enumerate(results)])

    prompt = f"""
    You are an expert customer support agent specializing in Tata Motors car models. You are highly knowledgeable about the car manuals and provide detailed, accurate solutions to customer queries. 

    A customer has asked the following query:
    
    User's Query: {query}
    
    Below are relevant pieces of information from the car manual for the specified car model(s) that can help you answer the query:

    Context:
    {context}

    Using the context provided, respond with the most helpful and precise answer possible. You should act as an expert in Tata Motors cars and be as informative as possible.
    """

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.2:latest", prompt],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error from Ollama CLI: {result.stderr.strip()}"
    except Exception as e:
        return f"An error occurred while querying the LLM: {e}"

# Function to create Gradio interface
def create_gradio_interface():
    def process_query(car_model, query):
        try:
            results = similarity_search(query=query, car_model=car_model)
            if results:
                formatted_results = [
                    [
                        result['score'],
                        result['car_model'],
                        result['chunk_index'],
                        result['source_file'],
                        result['chunk_text']
                    ]
                    for result in results
                ]

                llm_response = query_to_llm_with_ollama(query, results)
                
                # Print all results in the terminal
                print("\nAll Results:")
                for idx, result in enumerate(results, start=1):
                    print(f"\nResult {idx}:")
                    print(f"Score: {result['score']}")
                    print(f"Car Model: {result['car_model']}")
                    print(f"Chunk Index: {result['chunk_index']}")
                    print(f"Source File: {result['source_file']}")
                    print(f"Chunk Text: {result['chunk_text']}")

                print("\nLLM Response:")
                print(llm_response)

                return llm_response, formatted_results
            else:
                return "No results found for your query.", []
        except Exception as e:
            return f"An error occurred: {e}", []

    # Gradio UI components
    with gr.Blocks(css="""
        .submit-btn { width: 130px;  margin: 0 auto;}
    """) as interface:
        gr.Markdown(
            """
            <h1 style="text-align: center;">DriveXpert AI Assistant</h1>
            <p style="text-align: center;">Tata DriveXpert lets users quickly solve their car-related queries using AI-powered insights tailored to their specific Tata model.</p>
            """
        )
        with gr.Row():
            car_model_dropdown = gr.Dropdown(
                choices=['Altroz', 'Harrier', 'Nexon-ev', 'Nexon', 'Punch-ev', 'Punch', 'Safari', 'Tiago', 'Tigor', 'Zest'],
                label="Select Car Model"
            )

            query_input = gr.Textbox(
                label="Enter your query",
                lines=1
            )

        submit_button = gr.Button("Submit", elem_classes="submit-btn")

        llm_output = gr.Textbox(
            label="LLM Response",
            interactive=False,
            lines=7,
            max_lines=7
        )

        results_table = gr.Dataframe(
            headers=["Score", "Car Model", "Chunk Index", "Source File", "Chunk Text"],
            label="Search Results",
            interactive=False
        )

        submit_button.click(
            process_query,
            inputs=[car_model_dropdown, query_input],
            outputs=[llm_output, results_table]
        )

    interface.launch()

# Run the Gradio interface
if __name__ == "__main__":
    create_gradio_interface()



'''

import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import subprocess

# Initialize Qdrant client
collection_name = 'car_manuals'
qdrant_url = "http://localhost:6333"
client = QdrantClient(url=qdrant_url)

# Set up embeddings model
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Function to perform similarity search
def similarity_search(query: str, car_model: str, top_k: int = 3, score_threshold: float = 0.8):
    query_embedding = embeddings.embed_documents([query])[0]

    car_model_filter = Filter(
        must=[FieldCondition(key="car_model", match=MatchValue(value=car_model))]
    )

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        query_filter=car_model_filter
    )

    results = []
    for result in search_result:
        chunk_text = result.payload.get('chunk_text', 'No text available')
        results.append({
            'score': result.score,
            'car_model': result.payload.get('car_model', 'Unknown'),
            'chunk_index': result.payload.get('chunk_index', -1),
            'source_file': result.payload.get('source_file', 'Unknown'),
            'chunk_text': chunk_text
        })

    high_score_results = [r for r in results if r['score'] >= score_threshold]

    if not high_score_results and results:
        high_score_results = [max(results, key=lambda x: x['score'])]

    return high_score_results

# Function to query the LLM with Ollama
def query_to_llm_with_ollama(query, results):
    context = "\n".join([f"Chunk {i+1} (Car Model: {result['car_model']}): {result['chunk_text']}" 
                         for i, result in enumerate(results)])

    prompt = f"""
    You are an expert customer support agent specializing in Tata Motors car models. You are highly knowledgeable about the car manuals and provide detailed, accurate solutions to customer queries. 

    A customer has asked the following query:
    
    User's Query: {query}
    
    Below are relevant pieces of information from the car manual for the specified car model(s) that can help you answer the query:

    Context:
    {context}

    Using the context provided, respond with the most helpful and precise answer possible. You should act as an expert in Tata Motors cars and be as informative as possible.
    """

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.2:latest", prompt],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error from Ollama CLI: {result.stderr.strip()}"
    except Exception as e:
        return f"An error occurred while querying the LLM: {e}"

# Function to create Gradio interface
def create_gradio_interface():
    def process_query(car_model, query):
        try:
            results = similarity_search(query=query, car_model=car_model)
            if results:
                llm_response = query_to_llm_with_ollama(query, results)
                # Print results in terminal as well
                print("\nTop search results:")
                for idx, result in enumerate(results, start=1):
                    print(f"\nResult {idx}:")
                    print(f"Score: {result['score']}")
                    print(f"Car Model: {result['car_model']}")
                    print(f"Chunk Index: {result['chunk_index']}")
                    print(f"Source File: {result['source_file']}")
                    print(f"Chunk Text: {result['chunk_text']}")

                print("\nLLM Response:")
                print(llm_response)

                return llm_response, results
            else:
                return "No results found for your query.", []
        except Exception as e:
            return f"An error occurred: {e}", []

    # Gradio UI components
    car_model_dropdown = gr.Dropdown(
        choices=['Altroz', 'Harrier', 'Nexon-ev', 'Nexon', 'Punch-ev', 'Punch', 'Safari', 'Tiago', 'Tigor', 'Zest'],
        label="Select Car Model"
    )

    query_input = gr.Textbox(label="Enter your query")

    llm_output = gr.Textbox(label="LLM Response", interactive=False)
    chunks_output = gr.Dataframe(headers=["Score", "Car Model", "Chunk Index", "Source File", "Chunk Text"], interactive=False)

    # Gradio interface
    interface = gr.Interface(
        fn=process_query,
        inputs=[car_model_dropdown, query_input],
        outputs=[llm_output, chunks_output],
        analytics_enabled=False  # Disable analytics to avoid timeout
    )

    interface.launch()

# Run the Gradio interface
if __name__ == "__main__":
    create_gradio_interface()




'''






'''


from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import os
import subprocess

collection_name = 'car_manuals'
qdrant_url = "http://localhost:6333"
client = QdrantClient(url=qdrant_url)

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def similarity_search(query: str, car_model: str, top_k: int = 3, score_threshold: float = 0.8):
    query_embedding = embeddings.embed_documents([query])[0]

    car_model_filter = Filter(
        must=[FieldCondition(key="car_model", match=MatchValue(value=car_model))]
    )

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        query_filter=car_model_filter
    )

    results = []
    for result in search_result:
        chunk_text = result.payload.get('chunk_text', 'No text available')
        results.append({
            'score': result.score,
            'car_model': result.payload.get('car_model', 'Unknown'),
            'chunk_index': result.payload.get('chunk_index', -1),
            'source_file': result.payload.get('source_file', 'Unknown'),
            'chunk_text': chunk_text
        })

    high_score_results = [r for r in results if r['score'] >= score_threshold]

    if not high_score_results and results:
        high_score_results = [max(results, key=lambda x: x['score'])]

    return high_score_results

def query_to_llm_with_ollama(query, results):
    context = "\n".join([f"Chunk {i+1} (Car Model: {result['car_model']}): {result['chunk_text']}" 
                         for i, result in enumerate(results)])

    prompt = f"""
    You are an expert customer support agent specializing in Tata Motors car models. You are highly knowledgeable about the car manuals and provide detailed, accurate solutions to customer queries. 

    A customer has asked the following query:
    
    User's Query: {query}
    
    Below are relevant pieces of information from the car manual for the specified car model(s) that can help you answer the query:

    Context:
    {context}

    Using the context provided, respond with the most helpful and precise answer possible. You should act as an expert in Tata Motors cars and be as informative as possible.
    """

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.2:latest", prompt],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error from Ollama CLI: {result.stderr.strip()}"
    except Exception as e:
        return f"An error occurred while querying the LLM: {e}"

if __name__ == "__main__":
    query = input("Enter your query: ")
    car_model = input("Enter the car model name: ")

    try:
        results = similarity_search(query=query, car_model=car_model)
        if results:
            print("\nTop search results:")
            for idx, result in enumerate(results, start=1):
                print(f"\nResult {idx}:")
                print(f"Score: {result['score']}")
                print(f"Car Model: {result['car_model']}")
                print(f"Chunk Index: {result['chunk_index']}")
                print(f"Source File: {result['source_file']}")
                print(f"Chunk Text: {result['chunk_text']}")

            llm_response = query_to_llm_with_ollama(query, results)
            print("\nLLM Response:")
            print(llm_response)

        else:
            print("No results found for your query.")
    except Exception as e:
        print(f"An error occurred: {e}")


'''


'''
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import os

collection_name = 'car_manuals'
qdrant_url = "http://localhost:6333"

client = QdrantClient(url=qdrant_url)

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def similarity_search(query: str, car_model: str, top_k: int = 3):

    query_embedding = embeddings.embed_documents([query])[0]

    car_model_filter = Filter(
        must=[
            FieldCondition(
                key="car_model",
                match=MatchValue(value=car_model)
            )
        ]
    )

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        query_filter=car_model_filter
    )

    results = []
    for result in search_result:
        chunk_text = result.payload.get('chunk_text', 'No text available')
        results.append({
            'score': result.score,
            'car_model': result.payload.get('car_model', 'Unknown'),
            'chunk_index': result.payload.get('chunk_index', -1),
            'source_file': result.payload.get('source_file', 'Unknown'),
            'chunk_text': chunk_text
        })

    return results

if __name__ == "__main__":
    query = input("Enter your query: ")
    car_model = input("Enter the car model name: ")

    try:
        results = similarity_search(query=query, car_model=car_model)
        if results:
            print("\nTop search results:")
            for idx, result in enumerate(results, start=1):
                print(f"\nResult {idx}:")
                print(f"Score: {result['score']}")
                print(f"Car Model: {result['car_model']}")
                print(f"Chunk Index: {result['chunk_index']}")
                print(f"Source File: {result['source_file']}")
                print(f"Chunk Text: {result['chunk_text']}")
        else:
            print("No results found for your query.")
    except Exception as e:
        print(f"An error occurred: {e}")
'''