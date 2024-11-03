import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv

# Load DistilGPT-2 for text generation and DistilBERT for command matching
generation_model_name = "./fine_tuned_distilgpt2"
matching_model_name = "./fine_tuned_distilgpt2"

gen_tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(generation_model_name)

match_tokenizer = AutoTokenizer.from_pretrained(matching_model_name)
match_model = AutoModel.from_pretrained(matching_model_name)

# Load question-answer pairs from CSV file
qa_pairs = []
csv_file_path = "db.csv"  # Replace with the path to your CSV file

def load_qa_pairs_from_csv(csv_file_path):
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            if len(row) >= 2:  # Ensure there are at least two columns
                question, answer = row[0], row[1]  # Unpack only the first two columns
                qa_pairs.append({"question": question, "answer": answer})


# Call function to load QA pairs
load_qa_pairs_from_csv(csv_file_path)

# Precompute embeddings for the database questions
def get_embedding(text):
    inputs = match_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = match_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

question_embeddings = [get_embedding(pair["question"]) for pair in qa_pairs]

# Function to find the closest matching question in the database
def retrieve_answer(user_input, threshold=0.8):
    input_embedding = get_embedding(user_input).flatten()
    similarities = [cosine_similarity([input_embedding], [q_emb.flatten()])[0][0] for q_emb in question_embeddings]
    best_match_idx = int(np.argmax(similarities))

    # If the similarity is above the threshold, return the corresponding answer
    if similarities[best_match_idx] > threshold:
        return qa_pairs[best_match_idx]["answer"]
    else:
        return None  # No close match found

# Text generation fallback function
def generate_text(user_input):
    prompt = f"User: {user_input}\nAssistant:"
    inputs = gen_tokenizer.encode(prompt, return_tensors="pt")
    response_ids = gen_model.generate(
        inputs,
        max_length=len(inputs[0]) + 200,
        num_return_sequences=1,
        pad_token_id=gen_tokenizer.eos_token_id,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
    )
    response = gen_tokenizer.decode(response_ids[0], skip_special_tokens=True)
    assistant_response = response.split("Assistant:")[-1].strip()
    return assistant_response

# Chatbot loop
print("Hi! I am your lightweight assistant. How can I help you?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Attempt to retrieve an exact answer
    retrieved_answer = retrieve_answer(user_input)
    if retrieved_answer:
        print("Bot (Exact Match):", retrieved_answer)
    else:
        # Fallback to generation if no good match found
        response = generate_text(user_input)
        print("Bot (Conversation):", response)