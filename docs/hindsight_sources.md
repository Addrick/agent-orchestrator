# Hindsight Verified Sources

This document contains verified hard references for the Hindsight memory system.

## Official Documentation
- **Main Documentation**: [https://hindsight.vectorize.io](https://hindsight.vectorize.io)
- **Configuration Reference**: [https://hindsight.vectorize.io/developer/configuration](https://hindsight.vectorize.io/developer/configuration)
- **Retain Operation**: [https://hindsight.vectorize.io/developer/retain](https://hindsight.vectorize.io/developer/retain)
- **Reflect Operation**: [https://hindsight.vectorize.io/developer/reflect](https://hindsight.vectorize.io/developer/reflect)
- **Recall Operation**: [https://hindsight.vectorize.io/developer/retrieval](https://hindsight.vectorize.io/developer/retrieval)

## GitHub Repositories
- **Core Engine**: [https://github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight)
- **Cookbook & Examples**: [https://github.com/vectorize-io/hindsight-cookbook](https://github.com/vectorize-io/hindsight-cookbook)

## Technical Reference
- **Hindsight Research Paper**: [https://arxiv.org/abs/2512.12818](https://arxiv.org/abs/2512.12818)
- **API Schema (local copy)**: [architecture/external/hindsight_upstream_api.md](architecture/external/hindsight_upstream_api.md)
- **The live contract is the client, not this doc**: `src/memory/backend/hindsight.py` — it speaks
  v0.6.1+ (`/v1/default/banks/{id}/memories`). When the two disagree, the client wins.
