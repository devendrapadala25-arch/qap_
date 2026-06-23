import re
import json
import faiss
import numpy as np
import requests

from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer


# =====================================================
# MODELS
# =====================================================

@dataclass
class Block:
    type: str
    text: str
    level: int = 0
    line_no: int = 0


@dataclass
class Section:
    title: str
    level: int
    content: str = ""
    children: list = field(default_factory=list)


# =====================================================
# HEADING DETECTOR
# =====================================================

HEADING_KEYWORDS = {
    "introduction",
    "scope",
    "purpose",
    "qa matrix",
    "inspection requirements",
    "verification",
    "validation",
    "testing",
    "procedure",
    "responsibility",
    "references",
    "document control",
    "quality plan",
    "acceptance criteria"
}


def is_numbered_heading(line):

    return bool(
        re.match(
            r'^#*\s*\d+(?:\.\d+)*\s+.+$',
            line.strip()
        )
    )


def get_level(line):

    m = re.match(
        r'^#*\s*(\d+(?:\.\d+)*)',
        line.strip()
    )

    if m:
        return m.group(1).count(".") + 1

    return 1


def is_candidate_heading(line):

    line = line.strip()

    if not line:
        return False

    if line.startswith("-") or line.startswith("*"):
        return False

    if line.endswith(".") or line.endswith(":"):
        return False

    if len(line) > 60:
        return False

    words = line.split()

    if len(words) > 6:
        return False

    lower = line.lower()

    for keyword in HEADING_KEYWORDS:
        if keyword in lower:
            return True

    title_case_count = sum(
        1
        for w in words
        if w and w[0].isupper()
    )

    return title_case_count >= max(1, len(words)//2)


# =====================================================
# MARKDOWN PARSER
# =====================================================

def parse_markdown(text):

    blocks = []

    lines = text.splitlines()

    inside_table = False
    table = []

    for i, line in enumerate(lines):

        s = line.strip()

        if not s:
            continue

        if "<table" in s.lower():
            inside_table = True
            table = [s]
            continue

        if inside_table:

            table.append(s)

            if "</table>" in s.lower():

                blocks.append(
                    Block(
                        "table",
                        "\n".join(table),
                        0,
                        i
                    )
                )

                inside_table = False

            continue

        if is_numbered_heading(s):

            blocks.append(
                Block(
                    "heading",
                    s,
                    get_level(s),
                    i
                )
            )

            continue

        if is_candidate_heading(s):

            blocks.append(
                Block(
                    "candidate_heading",
                    s,
                    2,
                    i
                )
            )

            continue

        blocks.append(
            Block(
                "text",
                s,
                0,
                i
            )
        )

    return blocks


# =====================================================
# HIERARCHY BUILDER
# =====================================================

class HierarchyBuilder:

    def build_tree(self, blocks):

        roots = []
        stack = []

        for b in blocks:

            if b.type not in [
                "heading",
                "candidate_heading"
            ]:

                if stack:
                    stack[-1].content += b.text + "\n"

                continue

            level = (
                b.level
                if b.type == "heading"
                else 2
            )

            node = Section(
                title=b.text,
                level=level
            )

            while stack and stack[-1].level >= level:
                stack.pop()

            if stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)

            stack.append(node)

        return roots

    def build_retrieval_sections(
        self,
        roots
    ):

        sections = []

        def collect(node):

            text = node.content

            for child in node.children:

                text += (
                    "\n\n"
                    + child.title
                    + "\n"
                    + collect(child)
                )

            return text

        for root in roots:

            merged = Section(
                root.title,
                root.level
            )

            merged.content = collect(root)

            sections.append(merged)

        return sections


# =====================================================
# EMBEDDINGS
# =====================================================

class EmbeddingEngine:

    def __init__(self):

        self.model = SentenceTransformer(
            "BAAI/bge-m3"
        )

    def embed(self, text):

        return self.model.encode(
            text,
            normalize_embeddings=True
        )


# =====================================================
# FAISS
# =====================================================

class FAISSIndex:

    def __init__(self, dim):

        self.index = faiss.IndexFlatIP(dim)
        self.texts = []

    def add(self, embeddings, texts):

        self.index.add(
            np.array(embeddings)
            .astype("float32")
        )

        self.texts.extend(texts)

    def search(
        self,
        query_embedding,
        top_k=3
    ):

        q = np.array(
            [query_embedding]
        ).astype("float32")

        scores, idxs = self.index.search(
            q,
            top_k
        )

        results = []

        for s, i in zip(
            scores[0],
            idxs[0]
        ):

            if i == -1:
                continue

            results.append({
                "score": float(s),
                "text": self.texts[i]
            })

        return results


# =====================================================
# RETRIEVER
# =====================================================

class SectionRetriever:

    def __init__(self):

        self.embedder = EmbeddingEngine()
        self.index = None

    def build(self, sections):

        texts = []
        embs = []

        for s in sections:

            txt = (
                s.title
                + "\n"
                + s.content
            )

            texts.append(txt)

            embs.append(
                self.embedder.embed(txt)
            )

        dim = len(embs[0])

        self.index = FAISSIndex(dim)

        self.index.add(
            embs,
            texts
        )

        print(
            "Indexed sections:",
            len(texts)
        )

    def query(
        self,
        text,
        k=3
    ):

        q = self.embedder.embed(text)

        return self.index.search(
            q,
            top_k=k
        )


# =====================================================
# OLLAMA
# =====================================================

class Ollama:

    def __init__(
        self,
        model="qwen3:4b"
    ):
        self.model = model

    def generate(
        self,
        prompt
    ):

        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )

        return res.json()["response"]


# =====================================================
# VERIFIER
# =====================================================

class ChecklistVerifier:

    def __init__(
        self,
        retriever,
        llm
    ):

        self.retriever = retriever
        self.llm = llm

    def verify_item(self, item):

        results = self.retriever.query(
            item["section_name"],
            k=3
        )

        context = "\n\n".join(
            [
                r["text"]
                for r in results
            ]
        )

        prompt = f"""
You are a QAP auditor.

Checklist Item:
{item['question']}

Requirement:
{item['prompt']}

Context:
{context}

Return ONLY JSON:

{{
 "status":"PASS or FAIL",
 "confidence":95,
 "evidence":"...",
 "reason":"..."
}}
"""

        response = self.llm.generate(
            prompt
        )

        try:

            start = response.find("{")
            end = response.rfind("}")

            if start != -1:

                return json.loads(
                    response[start:end+1]
                )

        except:
            pass

        return {
            "status": "ERROR",
            "confidence": 0,
            "evidence": "",
            "reason": response
        }


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    with open(
        "sample_qap.md",
        "r",
        encoding="utf-8"
    ) as f:
        markdown = f.read()

    with open(
        "checklist.json",
        "r",
        encoding="utf-8"
    ) as f:
        checklist = json.load(f)

    blocks = parse_markdown(markdown)

    builder = HierarchyBuilder()

    tree = builder.build_tree(blocks)

    sections = builder.build_retrieval_sections(
        tree
    )

    retriever = SectionRetriever()

    retriever.build(sections)

    llm = Ollama(
        model="qwen3:4b"
    )

    verifier = ChecklistVerifier(
        retriever,
        llm
    )

    for item in checklist:

        print(
            "\n",
            "=" * 50
        )

        print(
            item["section_name"]
        )

        result = verifier.verify_item(
            item
        )

        print(
            json.dumps(
                result,
                indent=2
            )
        )