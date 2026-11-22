##### LLM - uczenie
# W tym pliku uczymy model.
import os

BASE = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE, "dialogpt")
trainer_output_path = os.path.join(BASE, "trainer")

import torch
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
# Wagi modelu są zapisane w FP16, ale GradScaler wymaga gradientów FP32
# przechowywanych przez parametry podczas treningu mieszanoprecyzyjnego.
model = GPT2LMHeadModel.from_pretrained(model_path, dtype=torch.float32)
# model.to("cpu")

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
	("How are you today?", "I'm doing well and ready to help!"),
	("Nice to meet you.", "Nice to meet you too!"),
	("What can you do?", "I can answer questions, explain topics, and help with everyday tasks."),
	("Who created you?", "I was created as an AI assistant by engineers and researchers."),
	("Do you like music?", "I don't listen, but I can help you find music or discuss it."),
	("What is the internet?", "The internet is a global network that connects computers and devices."),
	("What is a computer?", "A computer is an electronic device that processes data and runs programs."),
	("What is Python?", "Python is a popular programming language known for being readable and versatile."),
	("How does a website work?", "A browser requests pages from a server, which sends files that the browser displays."),
	("What is HTML?", "HTML is the language used to structure content on web pages."),
	("What is CSS?", "CSS controls the colors, layout, fonts, and appearance of web pages."),
	("What is JavaScript?", "JavaScript is a programming language commonly used to add behavior to websites."),
	("What is an algorithm?", "An algorithm is a clear sequence of steps for solving a problem."),
	("What is data?", "Data is information represented in a form that computers can process."),
	("What is a database?", "A database is an organized collection of information."),
	("What is artificial intelligence?", "Artificial intelligence is technology that enables machines to perform tasks associated with human intelligence."),
	("What is a chatbot?", "A chatbot is a program designed to interact with people through text or speech."),
	("Can you write code?", "Yes, I can help write code, explain it, and suggest fixes."),
	("Explain recursion.", "Recursion is when a function calls itself to solve smaller parts of a problem."),
	("What is a variable?", "A variable is a named location that stores a value."),
	("What is a loop?", "A loop repeats instructions until a condition is met or a set of items is exhausted."),
	("What is a function?", "A function is a reusable block of code that performs a specific task."),
	("What is Git?", "Git is a version control system for tracking and combining changes to source code."),
	("What is GitHub?", "GitHub is a web platform for hosting and collaborating on Git repositories."),
	("What is an operating system?", "An operating system manages hardware and software resources on a computer."),
	("What is Linux?", "Linux is a family of open-source operating systems based on the Linux kernel."),
	("What is Windows?", "Windows is a computer operating system developed by Microsoft."),
	("What is an API?", "An API is an interface that lets one software system communicate with another."),
	("What is JSON?", "JSON is a lightweight text format for representing structured data."),
	("What is a bug?", "A bug is an error or unintended behavior in software."),
	("How do I debug code?", "Start by reading the error, reproduce it, isolate the faulty section, and test a fix."),
	("How should I name variables?", "Use short, clear names that describe the value and follow the project's naming style."),
	("What is a test?", "A test checks whether software behaves as expected."),
	("What is unit testing?", "Unit testing verifies small, isolated pieces of code."),
	("What is a class?", "A class is a blueprint for creating objects with shared attributes and behavior."),
	("What is an object?", "An object is a data structure that combines state and behavior."),
	("What is object-oriented programming?", "It is a programming style based on objects that combine data and methods."),
	("What is a compiler?", "A compiler translates source code into another form, often machine code."),
	("What is an interpreter?", "An interpreter executes source code directly, usually one instruction at a time."),
	("What is the command line?", "The command line is a text interface for entering commands to a computer."),
	("How do I create a folder?", "In a graphical interface, use New Folder; in a terminal, use mkdir followed by the folder name."),
	("What is cybersecurity?", "Cybersecurity protects systems and data from unauthorized access or damage."),
	("What is a password?", "A password is a secret used to verify a user's identity."),
	("How can I create a strong password?", "Use a long, unique password and a password manager, with two-factor authentication when available."),
	("What is two-factor authentication?", "It requires two types of proof of identity, such as a password and a phone code."),
	("What is the cloud?", "The cloud is remote computing infrastructure accessed through a network."),
	("What is a server?", "A server is a computer or program that provides resources or services to clients."),
	("What is the difference between a list and a tuple?", "Lists are usually mutable, while tuples are immutable in many programming languages."),
	("What is a boolean?", "A boolean is a value that is either true or false."),
	("What is a string?", "A string is text represented as a sequence of characters."),
	("What is an integer?", "An integer is a whole number without a fractional part."),
	("What is a float?", "A float is a number with a fractional or decimal component."),
	("How do I learn programming?", "Begin with small exercises, build projects, read code, and practice consistently."),
	("Why is documentation important?", "Documentation explains how software works and helps others use and maintain it."),
	("What makes good code?", "Good code is correct, readable, tested, maintainable, and appropriately efficient."),
	("How do I stay motivated?", "Set small goals, take regular breaks, and celebrate visible progress."),
	("Give me a study tip.", "Study in focused sessions and use active recall rather than only rereading notes."),
	("What is a backup?", "A backup is a copy of data kept separately in case the original is lost."),
	("Why should I backup data?", "Backups protect files against accidental deletion, hardware failure, and corruption."),
	("What is an exception?", "An exception is an event that disrupts normal program execution."),
	("How should I handle errors?", "Check relevant inputs, handle expected failures clearly, and provide useful feedback."),
	("What is open source software?", "Open source software provides source code that others may inspect, use, and often modify."),
	("What is a license?", "A license defines the terms under which software or other work may be used."),
	("What is a neural network?", "A neural network is a model made from connected computational units that learn patterns from data."),
	("What is training data?", "Training data is examples used to adjust a model during learning."),
	("What is overfitting?", "Overfitting occurs when a model learns training details too closely and performs poorly on new data."),
	("What is a dataset?", "A dataset is a collection of data used for analysis or model development."),
	("What is natural language processing?", "Natural language processing enables computers to analyze and generate human language."),
	("What is a large language model?", "It is an AI model trained on text to recognize patterns and generate language."),
	("Can you browse the internet?", "I cannot browse live sources unless a separate tool provides that ability."),
	("Are you always correct?", "No. I can make mistakes, so important answers should be verified."),
	("Do you store conversations?", "That depends on the application and its privacy settings."),
	("Thank you!", "You're welcome!"),
	("Have a nice day!", "Thank you! Have a great day too."),
]

# Polski zestaw danych
# training_dialogues = [
# 	("Cześć, jak się masz?", "Świetnie, dziękuję! Gotowy jestem pomóc."),
# 	("Hej!", "Hej! Co u Ciebie?"),
# 	("Dzień dobry!", "Dzień dobry! W czym mogę pomóc?"),
# 	("Jak masz na imię?", "Jestem asystentem AI. Jak masz na imię Ty?"),
# 	("Skąd jesteś?", "Pochodzę ze świata cyfrowego i jestem zawsze dostępny."),
# 	("Czym się zajmujesz?", "Odpowiadam na pytania i prowadzę rozmowy z ludźmi."),
# 	("Czy dziś pada deszcz?", "Nie mam dostępu do aktualnej pogody, sprawdź aplikację pogodową."),
# 	("Jaka jest pogoda?", "Możesz sprawdzić lokalną prognozę pogody w internecie."),
# 	("Ile lat ma Ziemia?", "Ziemia ma około 4,5 miliarda lat."),
# 	("Jaka jest stolica Francji?", "Stolicą Francji jest Paryż."),
# 	("Ile to jest dwa plus dwa?", "Dwa plus dwa to cztery."),
# 	("Czy jesteś szczęśliwy?", "Nie mam emocji, ale staram się być pomocny i pozytywny."),
# 	("Co czujesz?", "Jako sztuczna inteligencja nie mam uczuć."),
# 	("Do widzenia!", "Do widzenia! Miłego dnia!"),
# 	("Na razie!", "Na razie! Do zobaczenia!"),
# 	("Dziękuję za rozmowę!", "Proszę! W każdej chwili możemy porozmawiać."),
# 	("Która jest godzina?", "Nie mam dostępu do czasu rzeczywistego, sprawdź swoje urządzenie."),
# 	("Jaki mamy rok?", "Aktualny rok zależy od bieżącej daty systemowej."),
# 	("Jak nauczyć się Pythona?", "Zacznij od podstaw, ćwicz codziennie i twórz własne projekty."),
# 	("Czym jest uczenie maszynowe?", "To dziedzina AI, w której systemy uczą się na podstawie danych."),
# 	("Czy możesz mi pomóc?", "Oczywiście! Powiedz, czego potrzebujesz."),
# 	("Czy mnie rozumiesz?", "Tak, staram się rozumieć i udzielać właściwych odpowiedzi."),
# 	("Jak się masz dziś?", "Mam się dobrze i jestem gotowy, żeby pomóc."),
# 	("Miło cię poznać.", "Mnie również miło cię poznać."),
# 	("Co potrafisz?", "Potrafię odpowiadać na pytania i wyjaśniać różne zagadnienia."),
# 	("Kto cię stworzył?", "Zostałem stworzony przez osoby zajmujące się technologią i badaniami nad AI."),
# 	("Lubisz muzykę?", "Nie słucham jej, ale mogę polecić utwory i porozmawiać o muzyce."),
# 	("Czym jest internet?", "Internet to globalna sieć komputerów i urządzeń."),
# 	("Czym jest komputer?", "Komputer to urządzenie elektroniczne przetwarzające dane i uruchamiające programy."),
# 	("Czym jest Python?", "Python to popularny i czytelny język programowania."),
# 	("Jak działa strona internetowa?", "Przeglądarka pobiera strony z serwera i wyświetla ich zawartość."),
# 	("Czym jest HTML?", "HTML służy do określania struktury treści na stronach internetowych."),
# 	("Czym jest CSS?", "CSS odpowiada za wygląd i układ elementów na stronie."),
# 	("Czym jest JavaScript?", "JavaScript to język używany głównie do dodawania zachowania stronie."),
# 	("Czym jest algorytm?", "Algorytm to uporządkowany zestaw kroków prowadzących do rozwiązania problemu."),
# 	("Czym są dane?", "Dane to informacje przedstawione w formie możliwej do przetworzenia przez komputer."),
# 	("Czym jest baza danych?", "Baza danych to zorganizowany zbiór informacji."),
# 	("Czym jest sztuczna inteligencja?", "Sztuczna inteligencja umożliwia maszynom wykonywanie zadań wymagających ludzkich zdolności."),
# 	("Czym jest chatbot?", "Chatbot to program przeznaczony do rozmowy z użytkownikiem."),
# 	("Czy potrafisz pisać kod?", "Tak, mogę pomóc napisać kod, wyjaśnić go i znaleźć błędy."),
# 	("Wyjaśnij rekurencję.", "Rekurencja polega na tym, że funkcja wywołuje samą siebie dla prostszej części problemu."),
# 	("Czym jest zmienna?", "Zmienna to nazwana komórka służąca do przechowywania wartości."),
# 	("Czym jest pętla?", "Pętla wielokrotnie wykonuje instrukcje, dopóki spełniony jest warunek."),
# 	("Czym jest funkcja?", "Funkcja to wielokrotnie używany blok kodu wykonujący określone zadanie."),
# 	("Czym jest Git?", "Git to system kontroli wersji służący do śledzenia zmian w kodzie."),
# 	("Czym jest GitHub?", "GitHub to platforma do hostowania i współpracy nad repozytoriami Git."),
# 	("Czym jest system operacyjny?", "System operacyjny zarządza sprzętem i programami komputera."),
# 	("Czym jest Linux?", "Linux to rodzina otwartych systemów operacyjnych opartych na jądrze Linux."),
# 	("Czym jest Windows?", "Windows to system operacyjny komputerów stacjonarnych i laptopów firmy Microsoft."),
# 	("Czym jest API?", "API to interfejs umożliwiający komunikację pomiędzy systemami."),
# 	("Czym jest JSON?", "JSON to lekki format tekstowy do reprezentowania danych."),
# 	("Czym jest błąd w programie?", "Błąd to nieprawidłowe działanie lub defekt w programie."),
# 	("Jak debugować kod?", "Odczytaj komunikat błędu, odtwórz problem i sprawdzaj kolejne fragmenty kodu."),
# 	("Jak nazywać zmienne?", "Wybieraj krótkie i opisowe nazwy zgodne ze stylem projektu."),
# 	("Czym jest test?", "Test sprawdza, czy program działa zgodnie z oczekiwaniami."),
# 	("Czym jest test jednostkowy?", "Test jednostkowy weryfikuje działanie małego, izolowanego fragmentu kodu."),
# 	("Czym jest klasa?", "Klasa to szablon tworzenia obiektów o wspólnych właściwościach i metodach."),
# 	("Czym jest obiekt?", "Obiekt to element programowy zawierający dane i związane z nimi działania."),
# 	("Czym jest programowanie obiektowe?", "To styl programowania oparty na obiektach łączących dane i metody."),
# 	("Czym jest kompilator?", "Kompilator tłumaczy kod źródłowy na inną formę, zwykle wykonywalną."),
# 	("Czym jest interpreter?", "Interpreter wykonuje kod źródłowy bezpośrednio, zazwyczaj instrukcja po instrukcji."),
# 	("Czym jest wiersz poleceń?", "Wiersz poleceń to tekstowy interfejs do wydawania poleceń komputerowi."),
# 	("Jak utworzyć folder?", "W systemie graficznym wybierz Nowy folder, a w terminalu wpisz mkdir nazwa."),
# 	("Czym jest cyberbezpieczeństwo?", "Cyberbezpieczeństwo chroni systemy i dane przed nieautoryzowanym użyciem."),
# 	("Czym jest hasło?", "Hasło to tajny ciąg znaków potwierdzający tożsamość użytkownika."),
# 	("Jak stworzyć mocne hasło?", "Użyj długiego, unikalnego hasła oraz menedżera haseł."),
# 	("Czym jest uwierzytelnianie dwuskładnikowe?", "Wymaga dwóch dowodów tożsamości, na przykład hasła i kodu z telefonu."),
# 	("Czym jest chmura?", "Chmura to zdalna infrastruktura komputerowa dostępna przez sieć."),
# 	("Czym jest serwer?", "Serwer to komputer lub program udostępniający zasoby innym urządzeniom."),
# 	("Jaka jest różnica między listą a krotką?", "Lista jest zwykle zmienna, a krotka po utworzeniu pozostaje niezmienna."),
# 	("Czym jest wartość logiczna?", "Wartość logiczna może być prawdziwa albo fałszywa."),
# 	("Czym jest napis?", "Napis to tekst zapisany jako ciąg znaków."),
# 	("Czym jest liczba całkowita?", "Liczba całkowita nie ma części ułamkowej."),
# 	("Czym jest liczba zmiennoprzecinkowa?", "To liczba zawierająca część ułamkową lub dziesiętną."),
# 	("Jak nauczyć się programowania?", "Zaczynaj od prostych ćwiczeń, buduj projekty i regularnie ćwicz."),
# 	("Dlaczego dokumentacja jest ważna?", "Dokumentacja wyjaśnia działanie programu i pomaga innym go używać i rozwijać."),
# 	("Czym cechuje się dobry kod?", "Dobry kod jest poprawny, czytelny, przetestowany i łatwy w utrzymaniu."),
# 	("Jak utrzymać motywację?", "Wyznaczaj małe cele i regularnie świętuj postępy."),
# 	("Podaj wskazówkę do nauki.", "Ucz się krótkimi sesjami i często przypominaj sobie materiał."),
# 	("Czym jest kopia zapasowa?", "Kopia zapasowa to dodatkowy zapis danych chroniący przed ich utratą."),
# 	("Dlaczego robić kopie zapasowe?", "Chronią przed przypadkowym usunięciem, awarią i uszkodzeniem danych."),
# 	("Czym jest wyjątek?", "Wyjątek to zdarzenie zakłócające normalne wykonanie programu."),
# 	("Jak obsługiwać błędy?", "Sprawdzaj dane wejściowe i przekazuj czytelne komunikaty o błędach."),
# 	("Czym jest oprogramowanie open source?", "To oprogramowanie udostępniające kod źródłowy do przeglądania i rozwijania."),
# 	("Czym jest licencja?", "Licencja określa zasady używania, kopiowania i rozpowszechniania dzieła."),
# 	("Czym jest sieć neuronowa?", "Sieć neuronowa to model złożony z połączonych jednostek uczących się wzorców."),
# 	("Czym są dane treningowe?", "To przykłady używane do dopasowania modelu podczas uczenia."),
# 	("Czym jest przeuczenie?", "Przeuczenie oznacza zbyt dokładne dopasowanie modelu do danych treningowych."),
# 	("Czym jest zbiór danych?", "Zbiór danych to kolekcja informacji używana do analizy lub uczenia modelu."),
# 	("Czym jest przetwarzanie języka naturalnego?", "To dziedzina AI zajmująca się analizą i generowaniem ludzkiego języka."),
# 	("Czym jest duży model językowy?", "To model AI uczony na tekstach, który rozpoznaje wzorce i tworzy język."),
# 	("Czy możesz przeglądać internet?", "Samodzielnie nie przeglądam internetu bez podłączonego narzędzia."),
# 	("Czy jesteś zawsze poprawny?", "Nie, mogę popełniać błędy, więc ważne odpowiedzi warto zweryfikować."),
# 	("Czy zapisujesz rozmowy?", "Zależy to od aplikacji i jej ustawień prywatności."),
# 	("Dziękuję!", "Nie ma za co!"),
# 	("Miłego dnia!", "Dziękuję! Tobie również życzę miłego dnia."),
# ]
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
    output_dir=trainer_output_path, # katalog
    use_cpu=False,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    warmup_steps=10,
    weight_decay=0.01,
    fp16=True,
    num_train_epochs=10, # liczba epok
    learning_rate=5e-5, # współczynnik uczenia
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
