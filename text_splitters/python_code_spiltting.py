from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
class Car:
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year

    def display_info(self):
        print(f"{self.year} {self.brand} {self.model}")


# Creating two objects from the class
car1 = Car("Toyota", "Corolla", 2022)
car2 = Car("Honda", "Civic", 2023)

car1.display_info()  # Output: 2022 Toyota Corolla
car2.display_info()  # Output: 2023 Honda Civic
"""

spiltter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=250,
    chunk_overlap=0,
)

chunks = spiltter.split_text(text)
print(len(chunks))
print(chunks[1])