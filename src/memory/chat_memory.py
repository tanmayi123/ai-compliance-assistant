from langchain.memory import ConversationBufferWindowMemory


def get_memory(k: int = 5):
    """
    Initialize and return a LangChain window memory object.
    k = number of recent exchanges to remember (default 5).
    """
    memory = ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        return_messages=True
    )
    return memory