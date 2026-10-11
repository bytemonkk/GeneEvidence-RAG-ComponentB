# GeneEvidence-RAG — Component B

A research project to build a gene–disease evidence retrieval system using
Retrieval-Augmented Generation (RAG).

## Current status

- [x] Selected and inspected a basic LangChain + ChromaDB RAG baseline.
- [x] Removed the original technology-company demo documents.
- [x] Removed the original demo vector index.
- [x] Configured Python 3.11 using uv.
- [ ] Install and validate dependencies.
- [ ] Load and inspect the TBGA dataset.
- [ ] Index gene–disease evidence with source metadata.
- [ ] Evaluate retrieval quality.
- [ ] Integrate the research model in a later phase.

## Technology stack

- Python 3.11
- uv
- LangChain
- ChromaDB
- Hugging Face embeddings and datasets

## Original baseline

This project began from
[KMmotors/Rag_System](https://github.com/KMmotors/Rag_System).

The original source scripts are retained as a baseline for adaptation.
The upstream repository's license and attribution terms must be verified
before redistribution.

## Research objective

Build a reproducible retrieval pipeline for gene–disease evidence using the
TBGA dataset. Preserve gene identifiers, disease identifiers, relationship
labels and stable source record IDs so that retrieved evidence can later be
connected to a separately trained relation-classification model.

## Current limitations

TBGA ingestion, gene-specific retrieval, evidence verification and evaluation
have not yet been implemented or validated.
