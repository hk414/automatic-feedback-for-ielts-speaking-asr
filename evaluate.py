# --------------------------------------------------------------
# evaluate.py – Simple Ragas evaluation (English only)
# --------------------------------------------------------------

import os
import json
import math
from typing import Dict

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy
)

# -------------------------------
# 1. Load .env (pip install python-dotenv)
# -------------------------------
from dotenv import load_dotenv
load_dotenv()

# -------------------------------
# 2. Evaluator Class
# -------------------------------
class IELTSFeedbackEvaluator:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found!\n"
                "→ Create a file named '.env' in this folder with:\n"
                "  OPENAI_API_KEY=sk-...\n"
                "→ Or run in PowerShell: $env:OPENAI_API_KEY='sk-...'"
            )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key,
        )
        self.embeddings = OpenAIEmbeddings(api_key=api_key)

        self.metrics = [
            Faithfulness(llm=self.llm),
            AnswerRelevancy(llm=self.llm, embeddings=self.embeddings)
        ]

    def load_text(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def prepare_dataset(self, human_txt: str, generated_txt: str) -> Dataset:
        return Dataset.from_dict({
            "question": ["Evaluate the IELTS Speaking performance based on the official band descriptors."],
            "answer": [generated_txt],
            "ground_truth": [human_txt],
            "contexts": [[human_txt]],
        })

    def evaluate_all(self, dataset: Dataset) -> Dict:
        try:
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings,
            )
            raw = result._repr_dict
            clean = {
                k: float(v) if isinstance(v, (int, float)) and not math.isnan(v) else None
                for k, v in raw.items()
            }
            return {
                "overall_metrics": clean,
                "detailed_results": dataset.to_list(),
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def save_results(results: Dict, path: str = "evaluation_results.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    ev = IELTSFeedbackEvaluator()

    human_txt = ev.load_text("human.txt")
    gen_txt   = ev.load_text("generated.txt")

    ds = ev.prepare_dataset(human_txt, gen_txt)
    res = ev.evaluate_all(ds)
    ev.save_results(res)

    if "error" in res:
        print("Evaluation FAILED:", res["error"])
    else:
        print("\n=== Ragas Evaluation Scores (English) ===")
        for name, score in res["overall_metrics"].items():
            print(f"{name:20}: {score:.3f}" if score is not None else f"{name:20}: N/A")


if __name__ == "__main__":
    main()