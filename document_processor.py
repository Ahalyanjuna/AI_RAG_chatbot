# Document Processing for Document RAG Chatbot

import os
import uuid
import PyPDF2
import docx
import csv
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

from PIL import Image
import cv2
import timm
import torch
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
from vector_db_manager import UnifiedEmbeddingAdapter, embedding_model


# Download required NLTK resources
#nltk.download('punkt', quiet=True)

efficientnet_model = timm.create_model("efficientnet_b0", pretrained=True)
efficientnet_model.eval()

unified_embedding_adapter = UnifiedEmbeddingAdapter(
    efficientnet_model, 
    target_dim=embedding_model.get_sentence_embedding_dimension()
)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model.eval()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# Normalization-free transform for BLIP
blip_transform = transforms.Compose([
    transforms.Resize((224, 224))
])

# Keep the original normalization transform for embeddings
image_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_image_for_embedding(file_path):
    """Load and preprocess image for embedding."""
    # Read image with OpenCV
    image = cv2.imread(file_path)
    if image is None:
        return "Error: Unable to read image"
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image for transformation
    pil_image = Image.fromarray(image)
    
    # Apply transform
    processed_image = image_transform(pil_image).unsqueeze(0)  # Add batch dimension
    return processed_image

def process_image_for_caption(file_path):
    """Load and preprocess image for captioning."""
    # Read image with OpenCV
    image = cv2.imread(file_path)
    if image is None:
        return "Error: Unable to read image"
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image for transformation
    pil_image = Image.fromarray(image)
    
    # Apply BLIP-specific transform (no normalization)
    processed_image = blip_transform(pil_image)
    return processed_image



def generate_caption(file_path):
    """Generate a caption for an image using BLIP Tiny."""
    image = process_image_for_caption(file_path)
    if isinstance(image, str):
        return image  # Return error message if image processing failed
    
    inputs = processor(image, return_tensors="pt")  # Prepare input
    with torch.no_grad():
        output = caption_model.generate(**inputs)  # Generate caption
    caption = processor.batch_decode(output, skip_special_tokens=True)[0]  # Convert to string
    
    return caption


def process_query_image(file_path):
    """
    Process an uploaded query image by generating:
    1. Image embedding
    2. Image caption
    3. Relevant details for querying
    """
    # Generate embedding
    embedding = get_image_embedding(file_path)
    if isinstance(embedding, str):  # Check for errors
        return {"status": "error", "message": embedding}

    # Generate caption
    caption = generate_caption(file_path)
    
    return {
        "status": "success",
        "embedding": embedding,
        "caption": caption
    }

def chunk_text(text, chunk_size=200, overlap=50):
    """Split text into overlapping chunks."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        
        if current_size + sentence_tokens > chunk_size and current_chunk:
            # Add current chunk to chunks
            chunks.append(' '.join(current_chunk))
            
            # Keep overlap sentences for next chunk
            overlap_size = 0
            overlap_chunk = []
            for s in reversed(current_chunk):
                s_tokens = len(s.split())
                if overlap_size + s_tokens <= overlap:
                    overlap_chunk.insert(0, s)
                    overlap_size += s_tokens
                else:
                    break
            
            current_chunk = overlap_chunk
            current_size = overlap_size
        
        current_chunk.append(sentence)
        current_size += sentence_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_pdf(file_path):
    """Extract text from PDF."""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    return text

def process_docx(file_path):
    """Extract text from DOCX."""
    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    
    return text

def process_csv(file_path):
    """Extract text from CSV by converting to descriptive text."""
    try:
        df = pd.read_csv(file_path)
        # Convert dataframe to readable text format
        text = f"CSV file with {len(df)} rows and {len(df.columns)} columns.\n"
        text += f"Columns: {', '.join(df.columns)}.\n\n"
        
        # Sample data summary
        text += "Sample data and statistics:\n"
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                text += f"Column {col}: min={df[col].min()}, max={df[col].max()}, "
                text += f"mean={df[col].mean():.2f}, median={df[col].median()}.\n"
            else:
                unique_vals = df[col].nunique()
                text += f"Column {col}: {unique_vals} unique values.\n"
                if unique_vals < 10:  # Only show all values if there are few
                    text += f"Values: {', '.join(df[col].unique().astype(str))}.\n"
                else:
                    text += f"Sample values: {', '.join(df[col].sample(5).astype(str))}.\n"
        
        # Add actual data rows as text
        text += "\nSample rows:\n"
        for i, row in df.head(5).iterrows():
            text += f"Row {i}: {' | '.join([f'{col}={val}' for col, val in row.items()])}.\n"
        
        return text
    except Exception as e:
        return f"Error processing CSV: {str(e)}"

def process_document(file_path):
    """Process any supported document type."""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        return process_pdf(file_path)
    elif file_ext == '.docx':
        return process_docx(file_path)
    elif file_ext == '.csv':
        return process_csv(file_path)
    elif file_ext in ['.jpg','.jpeg','.png']:
        return generate_caption(file_path)
    else:
        return f"Unsupported file type: {file_ext}"

def get_embeddings(texts):
    """Generate embeddings for a list of texts using the unified adapter."""
    return unified_embedding_adapter.forward_text(texts)

def get_image_embedding(file_path):
    """Generate image embedding."""
    image = process_image_for_embedding(file_path)
    if isinstance(image, str):
        return image  # Return error message if image processing failed
    
    with torch.no_grad():
        # Use the unified adapter to get the embedding
        embedding = unified_embedding_adapter.forward_image(image)
    
    return embedding

def add_image_to_knowledge_base(file_path, vector_db, vector_db_path, document_id=None):
    """Process image, generate embedding & caption, and store in vector DB."""
    if document_id is None:
        document_id = str(uuid.uuid4())

    # Get image embedding
    embedding = get_image_embedding(file_path)
    if isinstance(embedding, str):  # Check for errors
        return {"status": "error", "message": embedding}

    # Generate caption
    caption = generate_caption(file_path)
    
    # Minimal metadata - just store the caption separately
    metadata = {
        "doc_id": document_id,
        "type": "image",
        "caption": caption
    }

    # Add to vector database
    vector_db.add(
        embeddings=embedding, 
        texts="", 
        metadata_list=[metadata]
    )
    
    # Save the updated database
    vector_db.save(vector_db_path)
    
    return {
        "status": "success",
        "message": "Image added to knowledge base",
        "doc_id": document_id,
        "caption": caption
    }

def add_to_knowledge_base(file_path, vector_db, vector_db_path, document_id=None):
    """Process document and add to knowledge base."""
    if document_id is None:
        document_id = str(uuid.uuid4())
    
    # Handle images
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        return add_image_to_knowledge_base(file_path, vector_db, vector_db_path, document_id)
    
    # Extract text
    full_text = process_document(file_path)
    if full_text.startswith("Error") or full_text.startswith("Unsupported"):
        return {"status": "error", "message": full_text}
    
    # Chunk the text
    chunks = chunk_text(full_text)
    
    # Create metadata for each chunk
    metadata_list = [
        {
            "source": os.path.basename(file_path), 
            "doc_id": document_id, 
            "chunk_id": i,
            "type": "text"
        } for i in range(len(chunks))
    ]
    
    # Generate embeddings for chunks using vector_db_manager's get_embeddings
    from vector_db_manager import get_embeddings
    embeddings = get_embeddings(chunks)
    
    # Add to vector database
    vector_db.add(
        embeddings=embeddings, 
        texts=chunks, 
        metadata_list=metadata_list
    )
    
    # Save the updated database
    vector_db.save(vector_db_path)
    
    return {"status": "success", "message": f"Added {len(chunks)} chunks to knowledge base", "doc_id": document_id}