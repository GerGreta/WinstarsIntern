# src/ner_inference.py

# Список животных, соответствующих классам CV модели
ANIMALS = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

def extract_animals(text: str):
    """
    Извлекает названия животных из текста простым поиском по словарю ANIMALS
    """
    text_lower = text.lower()
    found_animals = [animal for animal in ANIMALS if animal in text_lower]
    return found_animals
