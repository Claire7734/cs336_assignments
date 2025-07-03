import regex
import os
from typing import BinaryIO
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from collections import Counter


def initialize_vocab(
        special_tokens_bytes: list[bytes]
) -> dict[int, bytes]:
    # chr(i).encode("utf-8") equals to bytes([i])
    vocab = {i :bytes([i]) for i in range(256)}
    curr_idx = 256
    for special_token in special_tokens_bytes:
        vocab[curr_idx] = special_token
        curr_idx += 1
    return vocab

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    special_tokens_bytes: list[bytes]
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    Adjusts boundaries to align with the next occurrence of any special token.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert all(isinstance(tok, bytes) for tok in special_tokens_bytes), (
        "All special tokens must be bytes"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    # mini_chunk might split across two consecutive mini_chunks
    # fix: overlapping reads slightly, carry the last len(split_special_token) - 1 bytes into the next read
    max_token_len = max(len(tok) for tok in special_tokens_bytes)
    overlap = max_token_len - 1

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        last_bytes = b""

        while True:
            file.seek(initial_position)
            chunk = file.read(mini_chunk_size)
            if chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            mini_chunk = last_bytes + chunk

            # Check all tokens
            found_pos = None
            for token in special_tokens_bytes:
                pos = mini_chunk.find(token)
                if pos != -1:
                    # Take the first occurrence of any token
                    if found_pos is None or pos < found_pos:
                        found_pos = pos

            if found_pos is not None:
                actual_offset = initial_position - len(last_bytes) + found_pos
                chunk_boundaries[bi] = actual_offset
                break

            # Prepare for next iteration
            if len(mini_chunk) >= overlap:
                last_bytes = mini_chunk[-overlap:]
            else:
                last_bytes = mini_chunk

            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(chunk, escaped, PAT):
    local_table = defaultdict(int)
    parts = regex.split("|".join(escaped), chunk)
    for part in parts:
        for word in regex.finditer(PAT, part):
            encoded = word.group().encode("utf-8")
            tuple_of_bytes = tuple(bytes([b]) for b in encoded)
            local_table[tuple_of_bytes] += 1
    return local_table

# def merge_tables(tables):
#     merged = defaultdict(int)
#     for table in tables:
#         for k, v in table.items():
#             merged[k] += v
#     return merged

def merge_tables(tables):
    return dict(sum((Counter(table) for table in tables), Counter()))

def pre_tokenization(
    input_path: str | os.PathLike,
    special_tokens_bytes: list[bytes],
    num_processes,
    parallelized = False,
) -> list[tuple[bytes], int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    special_tokens = [special_token_bytes.decode("utf-8", errors="ignore") for special_token_bytes in special_tokens_bytes]
    escaped = [regex.escape(tok) for tok in special_tokens]

    # The following is a serial implementation, but you can parallelize this 
    # by sending each start/end pair to a set of processes.
    # parallelized somehow slows down in test_train_bpe_special_tokens()
    pretoken_table = defaultdict(int)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, special_tokens_bytes)
        if parallelized:
            # somehow slower when using chunks
            chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            if parallelized:
                chunks.append(chunk)
            else:
                parts = regex.split("|".join(escaped), chunk)
                for part in parts:
                    words = regex.finditer(PAT, part)
                    for word in words:
                        encoded = word.group().encode("utf-8")
                        tuple_of_bytes = tuple(bytes([b]) for b in encoded)
                        pretoken_table[tuple_of_bytes] += 1

    if parallelized:
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_chunk, chunk, escaped, PAT) for chunk in chunks]
            results = [f.result() for f in futures]

        return list(merge_tables(results).items())
    else:
        return list(pretoken_table.items())

def merge_token_pair(pretokens_list: list[tuple[bytes], int],
        token_pairs: dict[tuple[bytes, bytes], int],
        pair_to_occurrences: dict[tuple[bytes, bytes], set],
        pair_to_merge: tuple[bytes, bytes],
):
    merged_pair = pair_to_merge[0] + pair_to_merge[1]
    affected_idxs = pair_to_occurrences.get(pair_to_merge, set())
    if not affected_idxs:
        return
    pair_to_occurrences.pop(pair_to_merge, None)

    for seq_idx in affected_idxs:
        tokens, count = pretokens_list[seq_idx]

        # # below code doesn't work for below corner case: 
        # # tokens = (b"A", b"B", b"B", b"B"), tokens = (b"A", b"B", b"B", b"B", b"B")
        # # pair_to_merge = (b"B", b"B")
        # new_tokens = []
        # i = 0
        # while i < len(tokens):
        #     if (
        #         i < len(tokens) - 1
        #         and (tokens[i], tokens[i + 1]) == pair_to_merge
        #     ):
        #         # Recompute affected neighboring pairs only (e.g., pairs around the merged ones)
        #         # update count in token_pairs for affected pair
        #         if i > 0:
        #             old_left = (new_tokens[-1], tokens[i])
        #             token_pairs[old_left] -= count
        #             pair_to_occurrences[old_left].discard(seq_idx)

        #             new_left = (new_tokens[-1], merged_pair)
        #             token_pairs[new_left] += count
        #             pair_to_occurrences[new_left].add(seq_idx)

        #         if i + 2 <  len(tokens):
        #             old_right = (tokens[i + 1], tokens[i + 2])
        #             token_pairs[old_right] -= count
        #             pair_to_occurrences[old_right].discard(seq_idx)

        #             new_right = (merged_pair, tokens[i + 2])
        #             token_pairs[new_right] += count
        #             pair_to_occurrences[new_right].add(seq_idx)

        #         new_tokens.append(merged_pair)
        #         i += 2
        #     else:
        #         new_tokens.append(tokens[i])
        #         i += 1

        # pretokens_list[seq_idx] = (tuple(new_tokens), count)
        
        ### inefficient correct code ###
        # --- Step 1: Remove old pairs ---
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            token_pairs[pair] -= count
            if token_pairs[pair] <= 0:
                del token_pairs[pair]
                pair_to_occurrences.pop(pair, None)
            else:
                pair_to_occurrences[pair].discard(seq_idx)
        # --- Step 2: Merge pair_to_merge ---
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair_to_merge:
                new_tokens.append(merged_pair)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_tokens = tuple(new_tokens)
        # --- Step 3: Add new pairs ---
        for i in range(len(new_tokens) - 1):
            pair = (new_tokens[i], new_tokens[i + 1])
            token_pairs[pair] += count
            pair_to_occurrences[pair].add(seq_idx)

        pretokens_list[seq_idx] = (new_tokens, count)


def bpe_merges(
        vocab: dict[int, bytes],
        vocab_size: int,
        pretokens_list: list[tuple[bytes], int],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges = []
    if len(vocab) >= vocab_size:
        return (vocab, merges)

    token_pairs = defaultdict(int)
    pair_to_occurrences = defaultdict(set)
    # 0: go through pretokens_list to create
    #   token_pairs & pair_to_occurrences
    for seq_idx, (tokens, count) in enumerate(pretokens_list):
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            token_pairs[pair] += count
            pair_to_occurrences[pair].add(seq_idx)

    while len(vocab) < vocab_size:
        # 1. most frequent pair, tie breaking
        pair_to_merge, count = max(token_pairs.items(), key=lambda x: (x[1], x[0]))
        # print(f"Merging pair {pair_to_merge} with count {count}, merges size: {len(merges)}")
        del token_pairs[pair_to_merge]
        merged_pair = pair_to_merge[0] + pair_to_merge[1]

        # 2. update vocab & merges
        vocab[len(vocab)] = merged_pair
        merges.append(pair_to_merge)

        # 3. pretokens_list: update with overlapped ones
        merge_token_pair(pretokens_list, token_pairs, pair_to_occurrences, pair_to_merge)

    return (vocab, merges)
