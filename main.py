import os
import asyncio
from openai import OpenAI
from dotenv import load_dotenv
from QwenIELTSEvaluator import QwenIELTSEvaluator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import whisper
from pathlib import Path
import whisper

from ragas.dataset_schema import SingleTurnSample
from datasets import Dataset 
from ragas.metrics import faithfulness, answer_correctness, answer_relevancy
from ragas import evaluate
from evaluate import IELTSFeedbackEvaluator

def save_txt(content: str, filepath: str):
    """Save content to text file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Saved: {filepath}")


if __name__ == "__main__":
    load_dotenv()
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    part_one = "https://raw.githubusercontent.com/hk414/audio-files/main/FrTPoIMqNFQ_5_5_part_1.mp3"
    part_two = "https://raw.githubusercontent.com/hk414/audio-files/main/FrTPoIMqNFQ_5_5_part_2.mp3"
    part_three = "https://raw.githubusercontent.com/hk414/audio-files/main/FrTPoIMqNFQ_5_5_part_3.mp3"
    feedback = "./testset/FrTPoIMqNFQ_5_5/FrTPoIMqNFQ_5_5_feedback.mp3"

    evaluator = QwenIELTSEvaluator(api_key=DASHSCOPE_API_KEY)
    
    # Track results and errors
    results = {}
    errors = {}
    
    # Evaluate Part 1
    print("\n" + "="*50)
    print("Evaluating Part 1...")
    print("="*50)
    try:
        p1_result = evaluator.evaluate_audio(part_one)
        print(p1_result)
        results['part1'] = p1_result
        save_txt(p1_result, "./testset/FrTPoIMqNFQ_5_5/part1_feedback.txt")
    except Exception as e:
        error_msg = f"Error in Part 1: {str(e)}"
        print(f"❌ {error_msg}")
        errors['part1'] = error_msg
        results['part1'] = None

    # Evaluate Part 2
    print("\n" + "="*50)
    print("Evaluating Part 2...")
    print("="*50)
    try:
        p2_result = evaluator.evaluate_audio(part_two)
        print(p2_result)
        results['part2'] = p2_result
        save_txt(p2_result, "./testset/FrTPoIMqNFQ_5_5/part2_feedback.txt")
    except Exception as e:
        error_msg = f"Error in Part 2: {str(e)}"
        print(f"❌ {error_msg}")
        errors['part2'] = error_msg
        results['part2'] = None

    # Evaluate Part 3
    print("\n" + "="*50)
    print("Evaluating Part 3...")
    print("="*50)
    try:
        p3_result = evaluator.evaluate_audio(part_three)
        print(p3_result)
        results['part3'] = p3_result
        save_txt(p3_result, "./testset/FrTPoIMqNFQ_5_5/part3_feedback.txt")
    except Exception as e:
        error_msg = f"Error in Part 3: {str(e)}"
        print(f"❌ {error_msg}")
        errors['part3'] = error_msg
        results['part3'] = None

    # Check if we have at least one successful result
    successful_parts = [k for k, v in results.items() if v is not None]
    
    if not successful_parts:
        print("\n" + "="*50)
        print("❌ ALL PARTS FAILED - Cannot proceed with evaluation")
        print("="*50)
        for part, error in errors.items():
            print(f"  {part}: {error}")
        exit(1)

    # Combine successful feedback
    print("\n" + "="*50)
    print(f"✅ Successfully evaluated: {', '.join(successful_parts)}")
    if errors:
        print(f"❌ Failed parts: {', '.join(errors.keys())}")
    print("="*50)

    # Build combined feedback from successful parts only
    feedback_parts = []
    if results['part1']:
        feedback_parts.append(f"Part 1:\n{results['part1']}")
    if results['part2']:
        feedback_parts.append(f"Part 2:\n{results['part2']}")
    if results['part3']:
        feedback_parts.append(f"Part 3:\n{results['part3']}")
    
    generated_feedback = "\n\n---\n\n".join(feedback_parts)
    
    # Save combined feedback
    save_txt(generated_feedback, "./testset/FrTPoIMqNFQ_5_5/combined_feedback.txt")

    # Load human feedback
    print("\n" + "="*50)
    print("Loading human feedback...")
    print("="*50)
    try:
        model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
        human_feedback = model.transcribe(feedback)
        save_txt(human_feedback, "./testset/FrTPoIMqNFQ_5_5/human_feedback.txt")
    except (FileNotFoundError, OSError):
        print("❌ Error: audio.mp3 not found")
        human_feedback = "No human feedback provided"

    # Run RAGAS evaluation
    print("\n" + "="*50)
    print("Running RAGAS Evaluation...")
    print("="*50)
    
    try:
        ev = IELTSFeedbackEvaluator()
        ds = ev.prepare_dataset(human_feedback, generated_feedback)
        res = ev.evaluate_all(ds)
        
        # Save results regardless of success/failure
        ev.save_results(res, "./testset/FrTPoIMqNFQ_5_5/evaluation_results.json")

        if res.get("status") == "failed" or "error" in res:
            print("\n❌ RAGAS Evaluation FAILED")
            print(f"Error: {res.get('error', 'Unknown error')}")
            if res.get("traceback"):
                print("\nTraceback:")
                print(res["traceback"])
        else:
            print("\n✅ RAGAS Evaluation Completed Successfully")
            print("\n=== Evaluation Scores ===")
            for name, score in res["overall_metrics"].items():
                if score is not None:
                    print(f"{name:25}: {score:.3f}")
                else:
                    print(f"{name:25}: N/A")
    
    except Exception as e:
        import traceback
        print(f"\n❌ Critical error during evaluation: {str(e)}")
        print("\nTraceback:")
        print(traceback.format_exc())
        
        # Save error information
        error_result = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "overall_metrics": {}
        }
        save_txt(
            str(error_result), 
            "./testset/FrTPoIMqNFQ_5_5/evaluation_error.json"
        )

    # Final summary
    print("\n" + "="*50)
    print("EXECUTION SUMMARY")
    print("="*50)
    print(f"✅ Successful parts: {len(successful_parts)}/3")
    if errors:
        print(f"❌ Failed parts: {len(errors)}/3")
        for part, error in errors.items():
            print(f"   - {part}: {error}")
    print("="*50)