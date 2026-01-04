import re
import copy

import torch


def qwen_vl_collate_fn(batch, tokenizer):
    """Collate function for Qwen3-VL."""    
    texts = []
    images = []
    
    for messages in batch:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        texts.append(text)

        # Extract images for the processor
        imgs = [c["image"] for c in messages[0]["content"] if c["type"] == "image"]
        images.append(imgs)

    inputs = tokenizer(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    # Return BOTH the processed tensors and the original raw messages
    return {"inputs": inputs, "raw_batch": batch}


def parse_yes_no(text):
    # 1. Isolate everything after the FIRST assistant tag (the model's turn)
    if "assistant" in text:
        # We split by "assistant" and join everything after the first one
        # to ensure we capture the 'Yes' and the explanation that follows.
        parts = text.split("assistant")
        actual_response = " ".join(parts[1:]).lower().strip()
    else:
        actual_response = text.lower().strip()

    # 2. Check for the specific word 'yes' or 'no'
    # prioritizing the start of the response
    words = actual_response.split()
    if not words: return -1

    # Check the first few words specifically
    first_few = words[:3]
    if any("yes" in w for w in first_few): return 1
    if any("no" in w for w in first_few): return 0

    # Fallback to general search if the model was wordy
    if "yes" in actual_response and "no" not in actual_response: return 1
    if "no" in actual_response and "yes" not in actual_response: return 0

    return -1


def retry_with_forced_prompt(model, tokenizer, original_message_list):
    # Deep copy to avoid mutating the original dataset batch
    retry_messages = copy.deepcopy(original_message_list)

    # Add the forceful instruction
    retry_messages.append({
        "role": "user",
        "content": "Answer with exactly one word: Yes or No."
    })

    text = tokenizer.apply_chat_template(retry_messages, tokenize=False, add_generation_prompt=True)
    imgs = [c["image"] for c in retry_messages[0]["content"] if c["type"] == "image"]

    inputs = tokenizer(
        text=[text],
        images=[imgs],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]