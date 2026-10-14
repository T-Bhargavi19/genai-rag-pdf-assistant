# GenAI RAG PDF Assistant


## Project Overview



A PDF Question Answering Assistant built using Retrieval-Augmented Generation (RAG). The application retrieves relevant passages from a PDF and uses a locally running Large Language Model (LLM) to generate context-based answers.



## Features



- Extracts text from PDF pages using PyMuPDF.

- Splits text into overlapping chunks.

- Generates embeddings using Sentence Transformers.

- Retrieves the top 3 matching chunks using cosine similarity.

- Uses a similarity threshold to filter potentially irrelevant questions.

- Generates answers using Llama 3.2 (3B) through Ollama.

- Displays answers with PDF page references and expandable source passages.



## Tech Stack



- Python

- Streamlit

- PyMuPDF

- NumPy

- Sentence Transformers (`all-MiniLM-L6-v2`)

- Ollama

- Llama 3.2 (3B)



## RAG Workflow



1. Extract text from PDF pages.

2. Split text into overlapping chunks.

3. Generate embeddings for the chunks.

4. Convert the user's question into an embedding.

5. Retrieve the top 3 matching chunks using cosine similarity.

6. Pass the retrieved context and question to the LLM.

7. Display the answer and source passages.



## Setup and Usage



### Prerequisites



- Python

- Ollama

- A local copy of the PDF used for testing



### Install Dependencies

```bash

pip install -r requirements.txt

```



### Download the LLM

```bash

ollama pull llama3.2:3b

```



### Run the Application

```bash

streamlit run app.py

```



Open the local URL displayed in the terminal.



## Current Scope and Limitations



- The current prototype processes PDF pages 12–18 of the configured textbook.

- Update `PDF\_PATH` in `app.py` to match the location of your local PDF.



## Future Improvements


- Add PDF upload support.

- Expand retrieval to larger documents.



\- Add PDF upload support.

\- Expand retrieval to larger documents.

