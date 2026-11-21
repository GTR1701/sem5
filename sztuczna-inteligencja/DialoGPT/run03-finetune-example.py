##### LLM - uczenie
# W tym pliku uczymy model.
import os

BASE = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE, "dialogpt")

import numpy as np
from transformers import (
	GPT2Tokenizer,
	GPT2LMHeadModel,
	TrainingArguments,
	Trainer,
	TrainerCallback,
)

# Załaduj tokenizer i model
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_path)
model.to("cpu")

# Przygotuj zbiór danych
dialoglist = []
training_dialogues = [
	("Hello, how are you?", "I'm doing great, thank you for asking!"),
	("Hi there!", "Hey! What's up?"),
	("Good morning!", "Good morning! How can I help you today?"),
	("What is your name?", "I'm an AI assistant. What's your name?"),
	("Where are you from?", "I'm from the digital world, always here to help!"),
	("What do you do?", "I answer questions and have conversations with people."),
	("Is it raining today?", "I don't have access to weather data, but you can check a weather app."),
	("What's the weather like?", "You could check your local weather forecast online."),
	("How old is the Earth?", "The Earth is approximately 4.5 billion years old."),
	("What is the capital of France?", "The capital of France is Paris."),
	("What is 2+2?", "2+2 equals 4."),
	("Are you happy?", "I try to be helpful and positive!"),
	("How do you feel?", "As an AI, I don't have feelings, but I'm here to assist you."),
	("Goodbye!", "Goodbye! Have a great day!"),
	("See you later!", "See you! Talk to you soon!"),
	("Thanks for the chat!", "You're welcome! Feel free to chat anytime!"),
	("What time is it?", "I don't have access to real-time information, but you can check your device."),
	("What year is it?", "It's 2024 as of my last update."),
	("How do I learn Python?", "You can start with online tutorials, practice coding daily, and build projects."),
	("What is machine learning?", "Machine learning is a field of AI where systems learn from data without explicit programming."),
	("Can you help me?", "Of course! I'll do my best to help. What do you need?"),
	("Do you understand me?", "Yes, I'm here to understand and assist you as best as I can."),
]

for question, answer in training_dialogues:
	dialoglist.append(question + " " + tokenizer.eos_token + answer + " " + tokenizer.eos_token)

# Zbiór do trenowania/testowania/walidacji jest ten sam
import datasets
ds =  datasets.Dataset.from_dict({"dialog": dialoglist})
dataset = datasets.DatasetDict({"train":ds, "validation":ds, "test":ds})

# Połącz wszystkie dialogi w jedno.
# Tutaj akurat zbędne ale potrzebne dla bardziej złożonych rozmów.
def concatenate_utterances(example):
    example['dialog'] = "".join(example['dialog'])
    return example
dataset = dataset.map(concatenate_utterances)

# Zakoduj zbiór danych
def encode(examples):
    encoded = tokenizer(examples['dialog'], truncation=True, padding='max_length', max_length=128)
    encoded['labels'] = encoded['input_ids'][:]
    # Ignoruj żetony wypełnienia nie licząc pierwszego (który jest też żetonem końca tekstu)
    encoded['labels'] = [[label for label in labels]
                         for labels in encoded['labels']]
    for i in range(len(encoded['labels'])):
        for j in range(len(encoded['labels'][i])):
            if (j > 0 and encoded['labels'][i][j] == tokenizer.pad_token_id and (encoded['labels'][i][j-1] == tokenizer.pad_token_id or encoded['labels'][i][j-1] == -100)):
                encoded['labels'][i][j] = -100
    return encoded

# Zastosuj 
encoded_dataset = dataset.map(encode, batched=True)

# Parametry trenowania
training_args = TrainingArguments(
    output_dir="trainer", # katalog
    use_cpu=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    warmup_steps=0,
    weight_decay=0.01,            
    fp16=False,
    num_train_epochs=2, # liczba epok          
    learning_rate=2e-4, # współczynnik uczenia
    logging_steps=1,
    logging_strategy="steps",
    disable_tqdm=False,
)

class TrainingProgressCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
    	print(
    		f"Postęp uczenia: krok {state.global_step}/{state.max_steps}, "
    		f"epoka {state.epoch:.2f}",
    		flush=True,
    	)

# Trener
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation'],
    callbacks=[TrainingProgressCallback()],
)

# Trenujemy model
trainer.train()

