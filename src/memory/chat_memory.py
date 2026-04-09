from langchain.memory import ConversationBufferMemory


def get_memory():
    """Initialize and return a LangChain conversation memory object."""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return memory