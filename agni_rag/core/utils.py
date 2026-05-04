import uuid


def stable_hash(text: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, text))
