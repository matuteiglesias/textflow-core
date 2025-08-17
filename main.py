from pathlib import Path
from snippetflow.ingest import ingest_paths
from pipeline.loader import load_clustered_nodes
from pipeline.polish import polish_clusters

paths = list(Path("ingest").glob("*.jsonl"))[:2]
vecs, nodes = ingest_paths(paths)

clusters = load_clustered_nodes(top_k=100)
summaries = polish_clusters(clusters, mode="summarize")

for i, s in enumerate(summaries):
    print(f"\n== Cluster {i} ==\n{s}\n")
