from ingestion import load_and_split_pdf
from vector_store import create_vector_store
from chunk_store import save_chunks


PDF_PATH = "data/business_communication.pdf"


print("Loading PDF...")

chunks = load_and_split_pdf(PDF_PATH)

print(f"Created {len(chunks)} chunks.")


print("Saving chunks...")

save_chunks(chunks)

print("Chunks saved successfully!")


print("Creating vector store...")

vector_store = create_vector_store(chunks)

print("Ingestion completed successfully!")