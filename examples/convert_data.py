#!/usr/bin/env python3
"""
Convert messages format data to prompt/answer format for EasyR1 framework.
"""
import json
import sys


def convert_messages_to_prompt_answer(input_file, output_file):
    """
    Convert messages format:
    {
      "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "images": [...]
    }

    To prompt/answer format:
    {
      "prompt": "...",
      "answer": "...",
      "images": [...]
    }
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    converted_data = []
    for item in data:
        if 'messages' not in item:
            print(f"Warning: Item without messages field, skipping: {item}", file=sys.stderr)
            continue

        messages = item['messages']

        # Extract user message (prompt) and assistant message (answer)
        user_message = None
        assistant_message = None

        for msg in messages:
            if msg['role'] == 'user':
                user_message = msg['content']
            elif msg['role'] == 'assistant':
                assistant_message = msg['content']

        if user_message is None or assistant_message is None:
            print(f"Warning: Missing user or assistant message, skipping: {item}", file=sys.stderr)
            continue

        converted_item = {
            'prompt': user_message,
            'answer': assistant_message,
        }

        # Copy over other fields (images, videos, etc.)
        for key, value in item.items():
            if key not in ['messages', 'prompt', 'answer']:
                converted_item[key] = value

        converted_data.append(converted_item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted_data)} items from {input_file} to {output_file}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file> <output_file>")
        sys.exit(1)

    convert_messages_to_prompt_answer(sys.argv[1], sys.argv[2])
