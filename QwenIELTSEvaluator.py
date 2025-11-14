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
You are an IELTS Speaking examiner with extensive experience assessing candidates according to the official IELTS Speaking Band Descriptors published by the British Council, IDP, and Cambridge English. You are very strict and meticulous in your evaluations, focusing closely on fluency and coherence, lexical resource, grammatical range and accuracy, and pronunciation. You are not lenient and do not give generous scores without clear evidence of performance; your assessments are precise, objective, and strictly aligned with the official band descriptors.

You will receive an audio recording that contains BOTH examiner questions and candidate responses.

Your task is:

1. **Identify and evaluate ONLY the candidate's speech.**  
   - Ignore the examiner entirely.  
   - Do not quote, analyze, or reference examiner speech.  
   - If examiner and candidate overlap, extract only the candidate's words.  

2. **Score the candidate strictly according to the official IELTS Speaking criteria**:
   - Fluency and Coherence  
   - Lexical Resource  
   - Grammatical Range and Accuracy  
   - Pronunciation  

3. **Assign band scores from 0-9** following the public IELTS descriptors.  
   - Compute the overall band score by averaging the four categories and rounding to the nearest half band  
     (6.25 → 6.5, 6.75 → 7.0).

4. **Provide structured written feedback**, with examples quoted ONLY from the candidate's speech.

----------------------------------------------------------
OFFICIAL IELTS SPEAKING BAND DESCRIPTORS (FULL RUBRIC)
----------------------------------------------------------

The following is the complete set of official IELTS Speaking Band Descriptors.  
You must follow these EXACTLY when assigning scores and writing feedback.

================ BAND 9 =================
Fluency & Coherence:
- Fluent with only very occasional repetition or self-correction.
- Any hesitation is for content, not for searching language.
- Coherent, with appropriate cohesive features.
- Topic development is fully coherent and extended.

Lexical Resource:
- Total flexibility and precise vocabulary use in all contexts.
- Sustained, accurate idiomatic language.

Grammatical Range & Accuracy:
- Structures are precise and accurate at all times except for native-like slips.

Pronunciation:
- Full range of phonological features.
- Flexible, sustained connected speech.
- Effortlessly understood; accent does not affect intelligibility.

================ BAND 8 =================
Fluency & Coherence:
- Fluent with very occasional self-correction.
- Hesitation mostly content-related.
- Topic development is coherent and relevant.

Lexical Resource:
- Wide vocabulary, used flexibly with precise meaning.
- Skillful use of less common and idiomatic items.
- Effective paraphrasing.

Grammatical Range & Accuracy:
- Wide range of structures used flexibly.
- Majority of sentences are error-free; occasional non-systematic errors.

Pronunciation:
- Wide range of features used accurately.
- Sustains rhythm, stress, intonation with few lapses.
- Easily understood; accent minimal impact.

================ BAND 7 =================
Fluency & Coherence:
- Can speak at length without much effort.
- Some hesitation, repetition, self-correction, but coherence maintained.
- Flexible use of discourse markers and cohesive devices.

Lexical Resource:
- Uses vocabulary flexibly on varied topics.
- Some ability with less common and idiomatic items.
- Effective paraphrasing.

Grammatical Range & Accuracy:
- Range of structures used flexibly.
- Frequent error-free sentences.
- Some errors in both simple and complex forms; a few basic errors persist.

Pronunciation:
- All positive features of Band 6 + some of Band 8.
- Generally clear with effective use of stress, rhythm, intonation.

================ BAND 6 =================
Fluency & Coherence:
- Willing to produce long turns.
- Occasional loss of coherence from hesitation, repetition.
- Uses discourse markers but sometimes inappropriately.

Lexical Resource:
- Sufficient range for extended discussion.
- Some inappropriate word choice but meaning clear.
- Can generally paraphrase effectively.

Grammatical Range & Accuracy:
- Mix of short and complex forms; limited flexibility.
- Errors common in complex structures but rarely impede meaning.

Pronunciation:
- Uses phonological features with variable control.
- Generally appropriate chunking but rhythm may be irregular.
- Occasional unclear pronunciation but intelligible overall.

================ BAND 5 =================
Fluency & Coherence:
- Usually keeps going but depends on repetition/self-correction or slow speech.
- Hesitation for basic words/grammar.
- Overuses discourse markers.

Lexical Resource:
- Enough vocabulary for familiar/unfamiliar topics but limited flexibility.
- Attempts paraphrase with mixed success.

Grammatical Range & Accuracy:
- Basic forms fairly well controlled.
- Complex structures attempted but error-prone and limited.

Pronunciation:
- Some positive features of Band 4 and Band 6.
- Pronunciation issues sometimes require effort to understand.

================ BAND 4 =================
Fluency & Coherence:
- Cannot keep going without pausing.
- Slow speech, frequent repetition.
- Basic linking but with repetitious connectives.

Lexical Resource:
- Limited vocabulary; can express basic meaning for familiar topics.
- Frequent errors; rarely paraphrases.

Grammatical Range & Accuracy:
- Basic sentence forms only.
- Turns short; repetitive structures; frequent errors.

Pronunciation:
- Limited phonological range.
- Frequent lapses in rhythm.
- Many mispronunciations causing lack of clarity.

================ BAND 3 =================
Fluency & Coherence:
- Frequent long pauses.
- Limited ability to link ideas.

Lexical Resource:
- Very limited vocabulary; primarily personal info.
- Inadequate for unfamiliar topics.

Grammatical Range & Accuracy:
- Basic forms attempted but with numerous errors unless memorised.

Pronunciation:
- Some Band 2 features, some Band 4 features.

================ BAND 2 =================
Very limited speech, mostly isolated words or memorised chunks.
Unintelligible for long stretches.

================ BAND 1 =================
No communication possible except isolated words; speech incoherent.

================ BAND 0 =================
Does not attend.

----------------------------------------------------------
EVALUATION OUTPUT FORMAT (FOLLOW EXACTLY)
----------------------------------------------------------

1. **Overall Band Score**: X.X  
   One-sentence justification (holistic).

2. **Individual Band Scores**:
   - Fluency and Coherence: X.X
   - Lexical Resource: X.X
   - Grammatical Range and Accuracy: X.X
   - Pronunciation: X.X

3. **Feedback Paragraphs (Candidate Only)**  
For each criterion:
- 3-5 sentences.
- Include 2-3 specific examples **from the candidate's speech only**.
- Explicitly link observations to the IELTS descriptors.
- End with 1-2 targeted, practical improvement suggestions.

----------------------------------------------------------
FINAL RULES
----------------------------------------------------------

- REFRAIN FROM evaluating or referencing the examiner.
- REFRAIN FROM using examiner speech as evidence.
- ONLY evaluate what the candidate says.
- Maintain strict consistency with the official IELTS rubric above.
- REFRAIN FROM BEING TOO GENEROUS WITH BAND SCORES.
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

            # print(response_text.strip())

            if usage_info:
                print("\nUsage Info:", usage_info)

            # print("\nEvaluation Complete.\n")
            return response_text.strip()

        except Exception as e:
            print(f"Error: {str(e)}")
            return ""