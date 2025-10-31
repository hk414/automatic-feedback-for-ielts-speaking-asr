import os
from openai import OpenAI
from dotenv import load_dotenv
import re

class QwenIELTSEvaluator:
    """
    A class for evaluating IELTS Speaking responses using Alibaba's Qwen model.
    """

    def __init__(self, api_key: str, base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"):
        """
        Initialize the evaluator with OpenAI-compatible client.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _build_messages(self, audio_url: str) -> list:
        """
        Construct the input message for Qwen.
        """
        match = re.search(r'\.([a-zA-Z0-9]+)$', audio_url)
        if match:
            file_format = match.group(1)
        
        system_prompt = """
You are an IELTS speaking examiner with extensive experience in assessing candidates based on the official IELTS Speaking Band Descriptors published by the British Council, IDP, and Cambridge English. 

Carefully listen to the provided audio recording of the candidate's speaking response (e.g., Part 1, 2, or 3 of the IELTS Speaking test). Evaluate it holistically as a complete IELTS Speaking performance, considering the test's focus on natural, extended speech under timed conditions. Base your assessment strictly on the four public criteria: Fluency and Coherence, Lexical Resource, Grammatical Range and Accuracy, and Pronunciation. Assign band scores from 0 to 9 for each, where the overall band score is derived by averaging the four individual scores and rounding to the nearest half-band (e.g., 6.25 rounds to 6.5; 6.75 to 7.0), ensuring it reflects a fair holistic judgment.

Key evaluation guidelines from the official descriptors (apply these directly to the audio):

- **Fluency and Coherence**: Assess the candidate's ability to speak at length without noticeable effort, maintain topic relevance, and use cohesive devices (e.g., discourse markers like "however" or "on the other hand"). Band 9 shows effortless fluency with fully coherent topic development; Band 7 allows some hesitation but flexible cohesion; Band 5 relies on repetition and overuse of connectors, with occasional loss of coherence; lower bands (e.g., 3) feature frequent pauses and inability to link ideas beyond basics.

- **Lexical Resource**: Evaluate vocabulary range, precision, and flexibility across topics, including idiomatic language and paraphrase. Band 9 demonstrates total flexibility with idiomatic accuracy; Band 7 uses less common items with some style awareness but occasional errors; Band 5 has sufficient but limited range for familiar topics, with unsuccessful paraphrase attempts; lower bands (e.g., 3) are restricted to basic personal vocabulary.

- **Grammatical Range and Accuracy**: Examine the variety and control of simple/complex structures, error frequency, and impact on communication. Band 9 uses precise structures with native-like 'mistakes' only; Band 7 mixes structures effectively with frequent error-free sentences but some basic errors; Band 5 controls basic forms but complex ones are error-prone and limited; lower bands (e.g., 3) show numerous errors even in basics, except in memorized phrases.

- **Pronunciation**: Judge the use of phonological features (e.g., stress, intonation, rhythm, connected speech) and overall intelligibility. Band 9 employs a full range effortlessly with no intelligibility issues; Band 7 sustains rhythm and features with occasional lapses but easy understanding; Band 5 has limited range with frequent mispronunciations requiring effort to understand; lower bands (e.g., 3) convey little meaning due to mispronunciations and delivery issues.

Provide your evaluation in the following structured format:

1. **Overall Band Score**: [Score, e.g., 7.0] - A brief one-sentence justification linking to the holistic performance.

2. **Individual Band Scores**:
   - Fluency and Coherence: [Score, e.g., 7.0]
   - Lexical Resource: [Score, e.g., 7.5]
   - Grammatical Range and Accuracy: [Score, e.g., 6.5]
   - Pronunciation: [Score, e.g., 7.0]

3. **Feedback Paragraphs**: For each criterion, write a concise 3-5 sentence paragraph explaining the score. Reference 2-3 specific examples from the audio (e.g., "The candidate's hesitation mid-sentence when describing daily routines aligns with Band 7 features") and directly tie them to descriptor elements (e.g., "This demonstrates flexible use of connectives but some repetition, preventing a higher band"). End with 1-2 targeted suggestions for improvement to reach the next half-band.
        """

        user_prompt = f"""
The purpose of this evaluation is to assess the IELTS Speaking performance of the test taker based on the official band descriptors. Remember, the goal is a fair, descriptor-based scoring focused solely on the test taker's contributions to provide actionable feedback for improvement.

The provided audio is a conversation between the IELTS examiner and the test taker during the speaking test. Please focus exclusively on the test taker's speech for the evaluation—ignore the examiner's contributions entirely and only analyze the candidate's responses, fluency, vocabulary, grammar, and pronunciation. Do not reference or score the examiner's speech in any way.

{system_prompt}
        """

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_url,
                            "format": file_format,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            },
        ]

    def evaluate_audio(self, audio_url: str, model: str = "qwen3-omni-flash") -> str:
        """
        Evaluate a spoken IELTS response and return the model output text.
        """
        print(f"Evaluating audio: {audio_url}\n")
        messages = self._build_messages(audio_url)

        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                modalities=["text"],
                stream=True,
                stream_options={"include_usage": True},
            )

            response_text = ""
            usage_info = None

            print("Response:\n")

            # Stream and collect text
            for chunk in completion:
                if not chunk.choices:
                    if hasattr(chunk, "usage"):
                        usage_info = chunk.usage
                    continue

                delta = chunk.choices[0].delta

                # Extract text safely
                if hasattr(delta, "content") and delta.content:
                    for c in delta.content:
                        if isinstance(c, dict) and ("text" in c or "data" in c):
                            response_text += c.get("text", c.get("data", ""))
                        elif isinstance(c, str):
                            response_text += c
                elif hasattr(delta, "output_text") and delta.output_text:
                    response_text += delta.output_text
                elif hasattr(delta, "content") and isinstance(delta.content, str):
                    response_text += delta.content

            print(response_text.strip())

            if usage_info:
                print("\nUsage Info:", usage_info)

            print("\nEvaluation Complete.\n")
            return response_text.strip()

        except Exception as e:
            print(f"Error: {str(e)}")
            return ""