import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

CHAT_NAMESPACE = "chat-history"


def _get_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME", "compliance-docs")
    return pc.Index(index_name)


def save_conversation(user_email: str, messages: list, lg_messages: list, title: str = "") -> str:
    """
    Save a conversation to Pinecone chat-history namespace.
    Returns the conversation ID.
    """
    if not messages:
        return ""

    index = _get_index()
    conversation_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    # Use first user message as title if not provided
    if not title:
        for msg in messages:
            if msg["role"] == "user":
                title = msg["content"][:60] + ("..." if len(msg["content"]) > 60 else "")
                break

    # Serialize messages
    metadata = {
        "user_email": user_email,
        "title": title,
        "timestamp": timestamp,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "messages": json.dumps(messages),
        "lg_messages": json.dumps([
            {"type": type(m).__name__, "content": m.content if hasattr(m, "content") else str(m)}
            for m in lg_messages
        ]),
        "message_count": len(messages),
    }

    # Use a dummy vector (zeros) — we're using Pinecone purely as a key-value store here
    dummy_vector = [0.0] * 1536

    index.upsert(
        vectors=[{
            "id": conversation_id,
            "values": dummy_vector,
            "metadata": metadata
        }],
        namespace=CHAT_NAMESPACE
    )

    return conversation_id


def update_conversation(conversation_id: str, messages: list, lg_messages: list):
    """Update an existing conversation in Pinecone."""
    if not conversation_id or not messages:
        return

    index = _get_index()

    # Fetch existing metadata to preserve original fields
    try:
        result = index.fetch(ids=[conversation_id], namespace=CHAT_NAMESPACE)
        existing = result.vectors.get(conversation_id)
        if not existing:
            return
        metadata = existing.metadata
    except Exception:
        return

    metadata["messages"] = json.dumps(messages)
    metadata["lg_messages"] = json.dumps([
        {"type": type(m).__name__, "content": m.content if hasattr(m, "content") else str(m)}
        for m in lg_messages
    ])
    metadata["message_count"] = len(messages)

    dummy_vector = [0.0] * 1536
    index.upsert(
        vectors=[{
            "id": conversation_id,
            "values": dummy_vector,
            "metadata": metadata
        }],
        namespace=CHAT_NAMESPACE
    )


def load_conversations(user_email: str, limit: int = 30) -> list:
    """
    Load all conversations for a user, sorted by timestamp descending.
    Returns list of dicts with id, title, timestamp, date, message_count.
    """
    index = _get_index()

    try:
        results = index.query(
            vector=[0.0] * 1536,
            top_k=limit,
            namespace=CHAT_NAMESPACE,
            filter={"user_email": {"$eq": user_email}},
            include_metadata=True
        )

        conversations = []
        for match in results.matches:
            conversations.append({
                "id": match.id,
                "title": match.metadata.get("title", "Untitled"),
                "timestamp": match.metadata.get("timestamp", ""),
                "date": match.metadata.get("date", ""),
                "message_count": match.metadata.get("message_count", 0),
            })

        # Sort by timestamp descending
        conversations.sort(key=lambda x: x["timestamp"], reverse=True)
        return conversations

    except Exception as e:
        print(f"[chat_history] Error loading conversations: {e}")
        return []


def load_conversation_messages(conversation_id: str) -> tuple:
    """
    Load full messages for a specific conversation.
    Returns (messages, lg_messages_raw) tuple.
    """
    index = _get_index()

    try:
        result = index.fetch(ids=[conversation_id], namespace=CHAT_NAMESPACE)
        vector = result.vectors.get(conversation_id)
        if not vector:
            return [], []

        messages = json.loads(vector.metadata.get("messages", "[]"))
        lg_messages_raw = json.loads(vector.metadata.get("lg_messages", "[]"))
        return messages, lg_messages_raw

    except Exception as e:
        print(f"[chat_history] Error loading conversation {conversation_id}: {e}")
        return [], []


def delete_conversation(conversation_id: str):
    """Delete a conversation from Pinecone."""
    index = _get_index()
    try:
        index.delete(ids=[conversation_id], namespace=CHAT_NAMESPACE)
    except Exception as e:
        print(f"[chat_history] Error deleting conversation {conversation_id}: {e}")


def group_by_date(conversations: list) -> dict:
    """Group conversations by relative date label."""
    from datetime import date, timedelta

    today = date.today()
    yesterday = today - timedelta(days=1)
    last_week = today - timedelta(days=7)

    groups = {"Today": [], "Yesterday": [], "Last 7 Days": [], "Older": []}

    for conv in conversations:
        try:
            conv_date = datetime.fromisoformat(conv["timestamp"]).date()
        except Exception:
            groups["Older"].append(conv)
            continue

        if conv_date == today:
            groups["Today"].append(conv)
        elif conv_date == yesterday:
            groups["Yesterday"].append(conv)
        elif conv_date >= last_week:
            groups["Last 7 Days"].append(conv)
        else:
            groups["Older"].append(conv)

    return {k: v for k, v in groups.items() if v}