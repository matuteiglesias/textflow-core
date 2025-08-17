import pathlib
from typing import List, Dict
from difflib import unified_diff

# Dummy placeholder
def summarize_nodes(nodes):
    return "\n\n".join(n.text for n in nodes)

def clean_text_basic(text: str) -> str:
    """Apply simple cleanup rules to text."""
    text = text.strip()
    text = text.replace("...", "…")
    while "  " in text:
        text = text.replace("  ", " ")
    return text

def polish_individual_nodes(nodes, ruleset="default", with_diff=False):
    """Apply per-node cleanup rules. Optionally return diffs."""
    cleaned = []
    diffs = []
    for n in nodes:
        orig = n.text
        new = clean_text_basic(orig)
        cleaned.append(new)
        if with_diff and orig != new:
            diff = "\n".join(unified_diff(orig.splitlines(), new.splitlines(), lineterm=""))
            diffs.append((n.metadata.get("header_path", "unknown"), diff))
    return cleaned, diffs if with_diff else cleaned

def polish_doc(nodes, strategy="rewrite"):
    """Rebuild the document using a polishing strategy."""
    if strategy == "rewrite":
        return "\n\n".join([clean_text_basic(n.text) for n in nodes])
    elif strategy == "summarize":
        return summarize_nodes(nodes)
    else:
        return "\n\n".join(n.text for n in nodes)

def polish_md_file(path: str, strategy="rewrite") -> str:
    """Polish a raw .md file (not structured nodes)."""
    md = pathlib.Path(path).read_text(encoding="utf-8")
    if strategy == "rewrite":
        return clean_text_basic(md)
    return md  # placeholder for other strategies

def polish_clusters(cluster_dict: Dict[int, List], mode="intro") -> Dict[int, str]:
    """Apply cluster-level polishing: currently only summarization."""
    output = {}
    for cluster_id, nodes in cluster_dict.items():
        if mode == "intro":
            summary = summarize_nodes(nodes)
            output[cluster_id] = summary
        else:
            output[cluster_id] = "\n\n".join(n.text for n in nodes)
    return output


# from polish import polish_individual_nodes

# # Suppose you have a list of nodes:
# cleaned, diffs = polish_individual_nodes(nodes, with_diff=True)

# for path, diff in diffs:
#     print(f"\n-- {path} --\n{diff}")
