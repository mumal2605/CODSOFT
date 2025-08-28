# app.py (Final Corrected Version)

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import tensorflow as tf
import logging
import numpy as np 

from model_utils import build_model, generate_text, get_latest_checkpoint, load_vocab

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Shakespearean Text Generator API",
    version="1.0.0"
)

# Global variables
model = None
char2idx = None
idx2char = None

class GenerateRequest(BaseModel):
    start_string: str = Field(..., example="ROMEO:")
    num_generate: int = Field(default=500, gt=0, le=2000)
    temperature: float = Field(default=0.8, gt=0.0, le=1.5)

class GenerateResponse(BaseModel):
    generated_text: str

@app.on_event("startup")
def load_model_on_startup():
    global model, char2idx, idx2char
    
    logger.info("API starting up. Loading model and vocabulary...")
    
    try:
        char2idx, idx2char, vocab_size = load_vocab()
        logger.info(f"Vocabulary and mappings loaded successfully. Vocab size: {vocab_size}")
    except FileNotFoundError:
        logger.error("vocab.json not found! Please run create_vocab.py first.")
        raise RuntimeError("vocab.json not found.")

    checkpoint_dir = './training_checkpoints'
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)

    if not latest_checkpoint:
        logger.error(f"No checkpoint file (.weights.h5) found in {checkpoint_dir}.")
        return

    embedding_dim = 256
    rnn_units = 1024

    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    
    try:
        model.load_weights(latest_checkpoint)
        logger.info(f"Model weights loaded successfully from: {latest_checkpoint}")
        model.summary(print_fn=logger.info)
    except Exception as e:
        logger.error(f"CRITICAL ERROR loading model weights: {e}")
        model = None

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check server logs.")

    if not all(char in char2idx for char in request.start_string):
        raise HTTPException(status_code=400, detail="Input contains characters not in vocabulary.")

    try:
        generated_text = generate_text(
            model=model,
            start_string=request.start_string,
            char2idx=char2idx,
            idx2char=idx2char, # <-- THE FIX IS HERE. Pass the variable directly.
            num_generate=request.num_generate,
            temperature=request.temperature
        )
        return GenerateResponse(generated_text=generated_text)
    except Exception as e:
        logger.error(f"Error during text generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Shakespearean Text Generator API. See /docs for usage."}