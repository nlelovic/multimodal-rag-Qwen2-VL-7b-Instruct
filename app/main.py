import os
import chainlit as cl
from byaldi import RAGMultiModalModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize models
def init_models():
    # Initialize Qwen model
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    
    # Initialize RAG model
    rag_model = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0")
    
    return model, tokenizer, rag_model

model, tokenizer, rag_model = init_models()

@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome to the Multimodal RAG Assistant! You can upload images or PDFs for indexing, and then ask questions about them.").send()

@cl.on_message
async def main(message: cl.Message):
    # Check for file uploads
    if message.elements:
        files = message.elements
        
        # Process and index uploaded files
        for file in files:
            file_path = file.path
            try:
                rag_model.index(
                    input_path=file_path,
                    index_name="user_docs",
                    store_collection_with_index=True,
                    overwrite=True
                )
                await cl.Message(content=f"Successfully indexed {file.name}").send()
            except Exception as e:
                await cl.Message(content=f"Error indexing {file.name}: {str(e)}").send()
        return

    # Process text query
    query = message.content
    try:
        # Search for relevant documents
        results = rag_model.search(query, k=3)
        
        # Format context from search results
        context = "\n".join([f"Document {r['doc_id']}, Page {r['page_num']}" for r in results])
        
        # Create prompt for Qwen
        prompt = f"""Context: {context}
        
Question: {query}

Please provide a detailed answer based on the context provided."""

        # Generate response with Qwen
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            num_return_sequences=1
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Send response
        await cl.Message(content=response).send()
        
    except Exception as e:
        await cl.Message(content=f"Error processing query: {str(e)}").send()

if __name__ == "__main__":
    cl.run()