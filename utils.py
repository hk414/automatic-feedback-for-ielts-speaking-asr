import os

def save_txt(content: str, filepath: str):
    """Save content to text file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved: {filepath}")