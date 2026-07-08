from langchain_text_splitters import RecursiveCharacterTextSplitter


text = """
The universe is everything that exists — all matter, energy, space, and time combined into one vast, expanding system. Studying it, a field called cosmology, tries to answer some of the biggest questions humans have ever asked: How did everything begin? How large is it? What is it made of? And where is it all headed? Scientists believe the universe began roughly 13.8 billion years ago in an event called the Big Bang, a period of extremely rapid expansion from an incredibly hot, dense state. Since then, space itself has been stretching, carrying galaxies farther apart from one another.
"""

spiltters = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)

result = spiltters.split_text(text)
print(len(result))
print(result)