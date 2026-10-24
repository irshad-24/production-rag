from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdf(file_path):

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks