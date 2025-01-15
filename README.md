# DriveXpert
Here's a `README.md` for your **DriveXpert** project:

---

# DriveXpert: AI-Powered Car Query Assistant

DriveXpert is an AI-powered assistant that helps car owners answer their queries using the car owner's manual. The system uses advanced AI techniques, including embeddings, similarity search, and a local LLM (Large Language Model), to provide relevant answers based on a Tata Motors car manual.

This project is built with various technologies such as **Qdrant**, **Ollama**, **Gradio**, **LangChain**, and **HuggingFace Embeddings**.

---

## Table of Contents

- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Flow](#project-flow)
- [Requirements](#requirements)
- [GitHub Repository](#github-repository)
- [License](#license)

---

## Introduction

DriveXpert allows users to select a Tata car model from a list and enter a query about the car. The system then searches the corresponding car model's manual and uses an AI model to generate an answer based on the retrieved content. This system empowers users with quick and precise information about their cars without needing human assistance.

---

## Technologies Used

- **Qdrant**: A vector database for storing and retrieving embeddings (vector representations of the text data from the car manuals).
- **Ollama**: A tool to run local LLM models (we use the `llama3.2:latest` model for query answering).
- **Gradio**: A Python library to create easy-to-use web interfaces.
- **LangChain**: A framework for building applications using language models.
- **HuggingFace Embeddings**: We use the HuggingFace model `BAAI/bge-large-en` for creating embeddings from text data.
- **Python**: The primary language used for the backend of this project.

---

## Installation

### Prerequisites

1. **Install Docker** for running Qdrant:
   - Download Docker from the official [Docker website](https://www.docker.com/products/docker-desktop).
   - Once installed, start Qdrant by running the following command:
     ```bash
     docker run -p 6333:6333 qdrant/qdrant
     ```

2. **Install Ollama**:
   - Download Ollama from [ollama.com](https://ollama.com) and follow the installation instructions.

3. **Install Python Libraries**:
   - Clone the GitHub repository and navigate into the project folder.
   - Install the required dependencies by running:
     ```bash
     pip install -r requirements.txt
     ```

---

## Usage

### Running the Application

1. Make sure **Qdrant** and **Ollama** are running.
2. Run the Python script to start the application:
   ```bash
   python app.py
   ```

   This will launch the Gradio interface in your browser.

3. **Interacting with the Interface**:
   - Select a car model from the dropdown (e.g., Altroz, Harrier, Nexon).
   - Enter a query related to the car model.
   - The system will process the query, retrieve relevant text from the car manual, and generate a response using the local AI model.

---

## Project Flow

1. **Select Car Model**: The user selects a Tata car model from the dropdown list.
2. **Enter Query**: The user types a query related to their car.
3. **Search in Car Manual**: The system searches the corresponding car manual using Qdrant for relevant chunks.
4. **Generate Answer**: The system sends the context (manual chunks) and query to Ollama's LLM to generate a detailed response.
5. **Display Answer**: The generated answer is displayed in the Gradio interface.

---

## Requirements

To run this project, you will need the following Python libraries:

- `gradio==3.35.0`
- `langchain==0.0.218`
- `qdrant-client==1.6.4`
- `huggingface-hub==0.13.3`
- `transformers==4.27.0`
- `torch==1.13.0`
- `requests==2.28.1`

You can install all the dependencies by running:

```bash
pip install -r requirements.txt
```

---

## GitHub Repository

You can find the code and detailed instructions in this [GitHub repository](https://github.com/username/repo_name).

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

Feel free to explore, modify, and contribute to the project! If you have any questions or suggestions, feel free to open an issue in the GitHub repository.

---

This `README.md` provides a simple explanation of the **DriveXpert** project, how to install and use it, and details about the technologies used.
