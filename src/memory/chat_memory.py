from langchain.memory import ConversationBufferWindowMemory


def get_memory(k: int = 5):
    """
    Initialize and return a LangChain window memory object.
    k = number of recent exchanges to remember (default 5).

    return_messages=False ensures chat_history is returned as a plain
    string, which is required by the ReAct PromptTemplate format.
    """
    memory = ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        return_messages=True  # must be True for OpenAI Functions agent (MessagesPlaceholder)
    )
    return memory