# Hugging Face calibration adapter

The adapter always requests `streaming=True` with the contract's exact dataset
revision and split. It ranks rows online with `sha256-rank-v1`, retaining only the
requested number of identities and text values, then tokenizes those rows in bounded
batches with the pinned tokenizer revision and remote code disabled. Dataset-sized
token arrays and full dataset downloads are outside the contract.

The canonical cache manifest includes the full dataset, preprocessing, tokenizer,
selection, sample-ID, content-digest, and token-ID records. It is serialized with
sorted compact JSON and atomically replaces a same-directory temporary file. Missing
optional dependencies, malformed streamed rows, conflicting retained IDs, and invalid
tokenizer batches fail explicitly without publishing a manifest.
