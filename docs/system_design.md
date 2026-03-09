# System Design

## Why two-stage recommendation
Two-stage recommendation balances relevance and latency. Candidate generation quickly narrows millions of items into a manageable set. Ranking then applies richer features and supervision to optimize engagement quality.

## Batch vs online responsibilities
Batch pipelines generate durable features and trained models. Online serving handles low-latency retrieval, reranking, and fallbacks.

## Cold-start handling
If a user has no interaction history, the API returns trending content by popularity. This reduces empty recommendations and protects user experience.

## Offline evaluation
The project computes Precision@K, Recall@K, and NDCG@K from held interactions. Operational metrics include candidate retrieval time and request latency.

## Deployment tradeoffs
Docker Compose is used for local iteration. Kubernetes manifests demonstrate production deployment shape for stateless API replicas and batch training jobs.

## Scalability considerations
The code is structured around interchangeable components. Local files can be replaced with feature stores, message queues, and managed model registries without API contract changes.
