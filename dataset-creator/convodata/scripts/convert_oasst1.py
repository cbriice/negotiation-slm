"""
Script to download OASST-1 dataset and convert to custom conversational format.
Filters conversations to 200-1000 tokens and samples ~7000 conversations.
Prioritizes highest-rated conversation paths per tree.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import random

from datasets import load_dataset
from tqdm import tqdm


def count_tokens_rough(text: str) -> int:
    """Rough token count estimation (words * 1.3)."""
    return int(len(text.split()) * 1.3)


def reconstruct_conversation_tree(messages_df) -> Dict[str, Dict[str, Any]]:
    """
    Reconstruct conversation trees from flat message structure.
    Returns dict mapping message_tree_id to tree data with messages and root_ids.
    """
    trees: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"messages": {}, "root_ids": []})

    for msg in messages_df:
        tree_id = msg["message_tree_id"]
        msg_id = msg["message_id"]
        parent_id = msg["parent_id"]

        trees[tree_id]["messages"][msg_id] = {
            "message_id": msg_id,
            "parent_id": parent_id,
            "text": msg["text"],
            "role": msg["role"],
            "lang": msg["lang"],
            "rank": msg.get("rank"),
            "review_result": msg.get("review_result", True),
        }

        if parent_id is None:
            trees[tree_id]["root_ids"].append(msg_id)

    return dict(trees)


def calculate_path_quality(conversation: List[Dict]) -> float:
    """
    Calculate quality score for a conversation path.
    Lower is better. Based on ranks and review results.
    """
    rank_sum = 0
    review_failures = 0

    for msg in conversation:
        rank = msg.get("rank")
        if rank is not None:
            rank_sum += rank

        if not msg.get("review_result", True):
            review_failures += 1

    avg_rank = rank_sum / len(conversation) if conversation else 0
    quality_score = avg_rank + (review_failures * 10)

    return quality_score


def extract_linear_conversations(tree_data: Dict[str, Any]) -> List[Tuple[List[Dict], float]]:
    """
    Extract linear conversation paths from a conversation tree.
    Returns list of (conversation_path, quality_score) tuples.
    """
    messages = tree_data["messages"]
    root_ids = tree_data["root_ids"]
    conversations = []

    def build_path(msg_id: str, current_path: List[Dict]):
        """Recursively build conversation paths."""
        if msg_id not in messages:
            return

        msg = messages[msg_id]
        current_path.append(msg)

        children = [m_id for m_id, m in messages.items() if m["parent_id"] == msg_id]

        if not children:
            quality = calculate_path_quality(current_path)
            conversations.append((current_path.copy(), quality))
        else:
            children_with_rank = [(c_id, messages[c_id].get("rank") if messages[c_id].get("rank") is not None else 999) for c_id in children]
            children_with_rank.sort(key=lambda x: x[1])

            for child_id, _ in children_with_rank:
                build_path(child_id, current_path.copy())

    for root_id in root_ids:
        build_path(root_id, [])

    return conversations


def convert_to_target_format(conversation: List[Dict], quality_score: float) -> Optional[Dict]:
    """
    Convert OASST-1 conversation to target format.
    Returns None if conversation doesn't meet criteria.
    """
    if len(conversation) < 2:
        return None

    total_text = " ".join([msg["text"] for msg in conversation])
    token_count = count_tokens_rough(total_text)

    if not (200 <= token_count <= 1000):
        return None

    messages = []

    for msg in conversation:
        messages.append({
            "role": msg["role"],
            "content": msg["text"]
        })

    return {
        "messages": messages,
        "system": {
            "assistant_style": "",
            "difficulty": "",
            "key_strategy": ""
        },
        "meta": {
            "opponent_style": "",
            "favorable_outcome": "",
            "deal_closed": "",
            "rounds": len(messages),
            "source": "oasst1",
            "lang": conversation[0]["lang"],
            "tree_id": conversation[0].get("message_id", "")[:8],
            "quality_score": quality_score
        }
    }


def main():
    print("Loading OASST-1 dataset from Hugging Face...")
    dataset = load_dataset("OpenAssistant/oasst1")

    train_data = dataset["train"]
    val_data = dataset["validation"]

    all_messages = list(train_data) + list(val_data)
    print(f"Loaded {len(all_messages)} messages")

    print("\nReconstructing conversation trees...")
    trees = reconstruct_conversation_tree(all_messages)
    print(f"Found {len(trees)} conversation trees")

    print("\nExtracting and ranking conversation paths per tree...")
    tree_best_paths: Dict[str, List[Tuple[List[Dict], float]]] = {}

    for tree_id, tree_data in tqdm(trees.items(), desc="Processing trees"):
        paths_with_quality = extract_linear_conversations(tree_data)
        paths_with_quality.sort(key=lambda x: x[1])
        tree_best_paths[tree_id] = paths_with_quality

    print("\nFiltering by token count and converting...")
    all_converted = []

    for tree_id, paths in tqdm(tree_best_paths.items(), desc="Converting"):
        for conversation, quality in paths:
            converted = convert_to_target_format(conversation, quality)
            if converted:
                all_converted.append(converted)

    print(f"Total valid conversations: {len(all_converted)}")

    all_converted.sort(key=lambda x: x["meta"]["quality_score"])

    english_only = [c for c in all_converted if c["meta"]["lang"] == "en"]
    print(f"English conversations: {len(english_only)}")

    target_count = 7000

    if len(english_only) >= target_count:
        print(f"\nSelecting top {target_count} highest-rated English conversations...")
        sampled = english_only[:target_count]
    else:
        print(f"\nNot enough English conversations ({len(english_only)}). Using all languages...")
        if len(all_converted) >= target_count:
            sampled = all_converted[:target_count]
        else:
            print(f"Warning: Only {len(all_converted)} conversations available")
            sampled = all_converted

    random.shuffle(sampled)

    output_path = Path(__file__).parent.parent / "data" / "oasst1-conversational.jsonl"
    print(f"\nWriting to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        for convo in sampled:
            f.write(json.dumps(convo, ensure_ascii=False) + "\n")

    print(f"✓ Successfully wrote {len(sampled)} conversations to {output_path}")

    print("\n=== Statistics ===")
    token_counts = [count_tokens_rough(" ".join([m["content"] for m in c["messages"]]))
                    for c in sampled]
    round_counts = [c["meta"]["rounds"] for c in sampled]
    quality_scores = [c["meta"]["quality_score"] for c in sampled]

    print(f"Token count - Min: {min(token_counts)}, Max: {max(token_counts)}, "
          f"Avg: {sum(token_counts)/len(token_counts):.1f}")
    print(f"Rounds - Min: {min(round_counts)}, Max: {max(round_counts)}, "
          f"Avg: {sum(round_counts)/len(round_counts):.1f}")
    print(f"Quality score - Min: {min(quality_scores):.2f}, Max: {max(quality_scores):.2f}, "
          f"Avg: {sum(quality_scores)/len(quality_scores):.2f}")

    langs = defaultdict(int)
    for c in sampled:
        langs[c["meta"]["lang"]] += 1
    print(f"Languages: {dict(sorted(langs.items(), key=lambda x: x[1], reverse=True))}")


if __name__ == "__main__":
    random.seed(42)
    main()
