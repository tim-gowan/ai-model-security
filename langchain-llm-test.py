from langchain_ollama import ChatOllama

model = ChatOllama(
    model="gpt-oss:20b",
    validate_model_on_init=True,
    temperature=0.8,
    num_predict=256,
    # other params ...
)

messages = [
    ("system", "You are a helpful translator. Translate the user sentence to Bahasa Indonesia."),
    ("human", "I love programming."),
]
model.invoke(messages)


stream = model.stream(messages)
full = next(stream)
for chunk in stream:
    full += chunk
    print(chunk.content, end="", flush=True)
