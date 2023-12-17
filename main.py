from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

vectorstore = DocArrayInMemorySearch.from_texts(
    ["MTG is a collectable card game",
     "Each player starts the game with 20 life",
     "Get your opponent's 20 life points down to 0 and you win the game",
     "In MTG, a spell is any type of card cast by a player",
     "Spells are usually cast from your hand",
     "In special cases can be cast from other areas of the battlefield like your library or your graveyard",
     "Land cards are the only type of card that is not considered a spell",
     "Creature cards in Magic serve as your primary means to attack, defend, and activate abilities during your turn",
     "They come in a wide array of shapes, sizes, and varying levels of power, so the damage dealt and received by your creatures will depend on these factors",
     "Creatures can't attack the same turn they enter the battlefield; this is known as 'summoning sickness'"],
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Refuse to answer questions that are not related to magic the gathering/MTG

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt | model | output_parser

print("Enter your MTG questions:")
user_input = input()
output = chain.invoke(user_input)
print(output)
