# Query Engine for Document RAG Chatbot

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    pipeline
)
from vector_db_manager import get_query_embedding

# Use a smaller model and optimize memory usage
device = torch.device("cpu")  # Force CPU to avoid CUDA memory issues

# LLM for text generation - Use base model instead of large
tokenizer_llm = AutoTokenizer.from_pretrained("google/flan-t5-base")
model_llm = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base", 
    torch_dtype=torch.float16  # Reduce precision to save memory
).to(device)

# QA model - Use a smaller model
qa_tokenizer = AutoTokenizer.from_pretrained("deepset/deberta-v3-large-squad2")
qa_model = AutoModelForQuestionAnswering.from_pretrained(
    "deepset/deberta-v3-large-squad2", 
    torch_dtype=torch.float16
).to(device)

# Use CPU for pipeline
qa_pipeline = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer, device=-1)

# Rest of the code remains the same as in the original file
def generate_answer(query, context_texts, max_length=5000):
    """Generate an answer using the LLM, encouraging list outputs when appropriate."""
    prompt = f"""
    Based on the following information, please answer the question.

    Question: {query}

    Information:
    {" ".join(context_texts)}

    If the answer is a list, provide all relevant items clearly and completely.

    Answer:
    """

    inputs = tokenizer_llm(prompt, return_tensors="pt", max_length=5000, truncation=True).to(device)

    outputs = model_llm.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=4,            # Reduced from 7 to save computation
        temperature=0.4,        # Lower for factual accuracy
        repetition_penalty=1.2, # Reduce repetition
        do_sample=True, 
        early_stopping=True
    )

    return tokenizer_llm.decode(outputs[0], skip_special_tokens=True)


# In query_engine.py, modify query functions to prioritize captions
def query_knowledge_base(query, vector_db, top_k=5):
    query_embedding = get_query_embedding(query)
    results = vector_db.search(query_embedding, k=top_k)

    if not results:
        return {
            "answer": "I don't have enough information to answer that question.",
            "confidence": None,
            "sources": [],
            "contexts": []
        }

    # Separate text and image results
    image_results = [r for r in results if 'caption' in r['metadata']]
    text_results = [r for r in results if 'caption' not in r['metadata']]

    context_texts = []
    sources = []

    # If query seems image-related, prioritize image captions
    image_related_keywords = ['image', 'picture', 'photo', 'snapshot', 'visual']
    is_image_query = any(keyword in query.lower() for keyword in image_related_keywords)

    if is_image_query and image_results:
        # Return the captions of the most relevant images
        context_texts = [f"Image description: {r['metadata']['caption']}" for r in image_results]
        sources = [r['metadata'].get('doc_id', 'unknown') for r in image_results]
        
        # If only image captions, generate a summary
        answer = generate_answer(query, context_texts)
    else:
        # Regular text-based query processing
        if text_results:
            context_texts = [r['text'] for r in text_results]
            sources = [r['metadata'].get('source', 'unknown') for r in text_results]
            
            # Use QA model or generate answer
            try:
                context = context_texts[0]
                qa_result = qa_pipeline(question=query, context=context)
                answer = qa_result['answer']
                confidence = qa_result['score']
            except Exception:
                answer = generate_answer(query, context_texts)
                confidence = None

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": list(set(sources)),
        "contexts": context_texts
    }


def query_knowledge_base_with_image(query, image_embedding, vector_db, top_k=5):
   
    """Search for the most similar image embeddings and return their captions.
    
    Args:
        query (str): Text query (will be ignored in this version)
        image_embedding (np.ndarray or torch.Tensor): Embedding of the input image
        vector_db (VectorDatabase): Vector database to search
        top_k (int): Number of top results to retrieve
    
    Returns:
        dict: Results with most similar image captions"""
    
    # Input validation
    if image_embedding is None:
        return {
            "answer": "No image embedding provided. Please ensure an image is uploaded.",
            "confidence": None,
            "sources": [],
            "contexts": []
        }
    
    # Ensure image_embedding is a numpy array
    try:
        if isinstance(image_embedding, torch.Tensor):
            image_embedding = image_embedding.numpy()
        
        # Normalize or handle multi-dimensional embeddings
        if image_embedding.ndim == 2:
            image_embedding = image_embedding[0]  # Take first embedding if 2D
    except Exception as e:
        return {
            "answer": f"Error processing image embedding: {str(e)}",
            "confidence": None,
            "sources": [],
            "contexts": []
        }
    
    # Perform vector search
    try:
        image_results = vector_db.search(image_embedding, k=top_k)
    except Exception as e:
        return {
            "answer": f"Error searching vector database: {str(e)}",
            "confidence": None,
            "sources": [],
            "contexts": []
        }
    
    # Extract captions and sources
    captions = [
        result['metadata'].get('caption', 'No caption available') 
        for result in image_results 
        if 'metadata' in result and 'caption' in result['metadata']
    ]
    
    sources = [
        result['metadata'].get('doc_id', 'unknown') 
        for result in image_results 
        if 'metadata' in result and 'doc_id' in result['metadata']
    ]
    
    # If no captions found
    if not captions:
        return {
            "answer": "",
            "confidence": None,
            "sources": [],
            "contexts": []
        }
    
    # Generate answer using the first (most similar) caption
    try:
        answer = generate_answer(
            "Describe the contents of this similar image", 
            [f"Image description: {captions[0]}"]
        )
    except Exception as e:
        answer = captions[0]  # Fallback to raw caption
    
    return {
        "answer": answer,
        "confidence": None,
        "sources": sources,
        "contexts": captions,
        "image_caption": captions[0]  # Most similar image caption
    }

