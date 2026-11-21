##### LLM - test
# W tym pliku testujemy wyuczony model.

model_path = "trainer\checkpoint-26"
tokenizer_path = "dialogpt"

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side = "right")
print("<< Tokenizer gotowy")

model = AutoModelForCausalLM.from_pretrained(model_path)
model.to("cpu")
print("<< Model gotowy")

query = input("Zapytanie: [enter - domyślne] ")
if query == "":
	query = "What is the numberwang? "
inputs = tokenizer.encode(query + tokenizer.eos_token, return_tensors='pt')
inputs.to("cpu")
print("<< Zapytanie gotowe")

outputs = model.generate(inputs, max_length=128, pad_token_id=tokenizer.eos_token_id)
outputs_extracted = outputs[:, inputs.shape[-1]:][0]
answer = tokenizer.decode(outputs_extracted, skip_special_tokens=True)
print("<< Odpowiedź gotowa")
print("Odpowiedź: {}".format(answer))
