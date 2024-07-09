# ResearchBuddy: Private RAG with Mistral 7B
This repository demonstrates a Retrieval-Augmented Generation (RAG) system using Langchain, Pathway, and the Mistral 7B Instruct model for efficient querying of documents stored in a vector database. The system is set up in a Google Colab environment and integrates a public Pathway Vector Store server.
# Problem and Solution:
The aim of this project is to build a chatbot that can help users with specific queries by retrieving and generating responses from a set of documents. This is particularly useful for scenarios where real-time, relevant, and context-aware answers are needed from a curated set of documents.

# Tech Stack
- Google Colab: For setting up and running the environment.
- Langchain: For creating the RAG pipeline.
- Pathway: For storing and retrieving document vectors.
- Public Pathway Vector Store: Utilizing a publicly available demo pipeline provided by Pathway.
- Mistral 7B Instruct: For generating responses to user queries.

# System Overview
- Document Loading: Load documents from a specified directory.
- Text Splitting: Split documents into chunks for efficient processing.
- Embedding: Use HuggingFace embeddings to convert text chunks into vectors.
- Vector Store: Use a public Pathway Vector Store for document storage and retrieval.
- Querying: Use Langchain to handle queries and retrieve relevant documents.

# Steps to Set Up and Run the System

**1. Set Up Google Colab Environment**
- Open Google Colab and mount Google Drive.
- Upload the necessary documents and models to Google Drive.

**2. Download Mistral 7B Instruct Model**
- Download the model from https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF.

**3. Install Required Packages in Colab**
- !pip install langchain
- !pip install -U langchain-community
- !pip install --prefer-binary pathway
- !pip install sentence_transformers
- !pip install huggingface-hub
- !pip install pypdf
- !pip -q install accelerate
- !pip install llama-cpp-python
- !pip -q install git+https://github.com/huggingface/transformers

**4. Load Documents and Create Vector Store**
    
    import pathway as pw
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
    
    #Load data
    data = pw.io.fs.read(
        "/content/drive/MyDrive/Data",
        format="binary",
        mode="streaming",
        with_metadata=True,
    )

    #Create embedder and splitter
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = CharacterTextSplitter()

**5.Initialize Pathway Vector Client**

    from langchain_community.vectorstores import PathwayVectorClient
    client = PathwayVectorClient(url="https://demo-document-indexing.pathway.stream")

**6. Create LlamaCpp LLM**

    from llama_cpp import LlamaCpp
    #Import Model
    llm = LlamaCpp(
        streaming=True,
        model_path="/content/drive/MyDrive/Model/mistral-7b-instruct-v0.1.Q4_K_M (1).gguf",
        temperature=0.75,
        top_p=1,
        verbose=True,
        n_ctx=4096
    )

**7. Query the Vector Store**

    query = "What is Pathway?"
    docs = client.similarity_search(query)
    print(docs)

**7. Create Interactive Loop for User Input**

    import sys
    from langchain.chains import RetrievalQA
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=client.as_retriever())
    
    while True:
        user_input = input("Input Prompt: ")
        if user_input == 'exit':
            print('Exiting')
            sys.exit()
        if user_input == '':
            continue
    
        result = qa.run(user_input)
        print(f"Answer: {result}")
    
**8. Additional Notes**
Ensure that the documents are correctly loaded and processed in Colab before querying the public Pathway Vector Store.
The system is designed to be flexible and can be adapted to various document types and query strategies.
This README provides a concise overview. For detailed setup instructions and troubleshooting, refer to the Pathway and Langchain documentation.

**9. Contributing**
Feel free to contribute by creating pull requests or raising issues.

# License
This project is licensed under the MIT License.

