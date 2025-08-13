import re
from typing import Optional


def post_process_response(answer: str, question: Optional[str] = "") -> str:
    """Clean up LLM output by stripping template artifacts, leaked instructions,
    ChatML tags, and other noisy patterns. Keeps the response concise and readable.

    This mirrors the production-grade sanitizer used by the legacy server
    implementation to ensure consistent behavior across v1/v2 stacks.
    """
    if not answer:
        return answer

    # Remove obvious template headings and debug sections
    template_patterns = [
        r"USER SITUATION:.*?(?=\n|$)",
        r"MEDICAL CONTEXT:.*?(?=\n|$)",
        r"CONVERSATION HISTORY:.*?(?=\n|$)",
        r"CRISIS RESPONSE PROTOCOL:.*?(?=\n|$)",
        r"SAFETY NOTICE:.*?(?=\n|$)",
        r"CRISIS\s*-\s*.*?(?=I understand|Let's|What|How|\.|$)",
        r"Assess immediate safety.*?accessible\.",
        r"{context}|{history}|{question}",
        r"^\s*User:.*?(?=\n|$)",
        r"^\s*Assistant asked:.*?(?=\n|$)",
        r"^\s*Assistant:.*?(?=\n|$)",
        r"^\s*Context:.*?(?=\n|$)",
        r"^\s*History:.*?(?=\n|$)",
    ]
    for pattern in template_patterns:
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE | re.MULTILINE)

    # Remove labels like Empathy:, Citation:, Follow-up question: etc
    answer = re.sub(
        r"(Empathy:|Citation:|Follow-up question:|User:|Assistant:|Context:|History:)",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    # Remove ChatML or similar special tokens
    answer = re.sub(r"<\|assistant\|\|?\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"<\|.*?\|>", "", answer, flags=re.IGNORECASE)

    # Remove internal instruction leaks and guideline headers
    internal_instruction_patterns = [
        r"INSTRUCTIONS?:.*?(?=\n|$)",
        r"INSTRUCTATIONS?:.*?(?=\n|$)",
        r"RESPONSE TEMPLATE:.*?(?=\n|$)",
        r"GUIDELINES:.*?(?=\n|$)",
        r"Start with empathy.*?(?=\n|$)",
        r"Cite ICD-11.*?(?=\n|$)",
        r"Ask 1 gentle.*?(?=\n|$)",
        r"Keep response.*?(?=\n|$)",
        r"Avoid generic.*?(?=\n|$)",
        r"Response Structure.*?(?=\n|$)",
        r"Formatting Guidelines.*?(?=\n|$)",
        r"Content Depth.*?(?=\n|$)",
        r"Professional Resource.*?(?=\n|$)",
        r"Balance Guidelines.*?(?=\n|$)",
        r"Question Type.*?(?=\n|$)",
        r"Quality Standards.*?(?=\n|$)",
        r"Personalization Elements.*?(?=\n|$)",
        r"Response Template.*?(?=\n|$)",
        r"1\. Empathy.*?(?=\n|$)",
        r"2\. Problem.*?(?=\n|$)",
        r"3\. Structured.*?(?=\n|$)",
        r"4\. Professional.*?(?=\n|$)",
        r"5\. Encouragement.*?(?=\n|$)",
        r"Crisis Questions.*?(?=\n|$)",
        r"How-to Questions.*?(?=\n|$)",
        r"Symptom Questions.*?(?=\n|$)",
        r"General Support.*?(?=\n|$)",
        r"Each piece of advice.*?(?=\n|$)",
        r"Include the reasoning.*?(?=\n|$)",
        r"Provide multiple options.*?(?=\n|$)",
        r"Ensure advice is.*?(?=\n|$)",
        r"Include appropriate.*?(?=\n|$)",
        r"Reference user.*?(?=\n|$)",
        r"Adapt advice based.*?(?=\n|$)",
        r"Consider cultural.*?(?=\n|$)",
        r"Provide age-appropriate.*?(?=\n|$)",
        r"Empathy Opening.*?(?=\n|$)",
        r"Problem Acknowledgment.*?(?=\n|$)",
        r"Structured Advice.*?(?=\n|$)",
        r"Professional Resources.*?(?=\n|$)",
        r"Encouragement Closing.*?(?=\n|$)",
        r"Acknowledge the user.*?(?=\n|$)",
        r"Let the user know.*?(?=\n|$)",
        r"Express empathy.*?(?=\n|$)",
        r"Understanding how.*?(?=\n|$)",
        r"EMPATHIC Connection:.*?(?=\n|$)",
        r"EMPHATIC Connection:.*?(?=\n|$)",
        r"EVIDENCE-BASED support:.*?(?=\n|$)",
    ]
    for pattern in internal_instruction_patterns:
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE | re.MULTILINE)

    # Trim excessive whitespace
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    answer = re.sub(r"\s{2,}", " ", answer)
    answer = answer.strip()

    # Ensure the output is not empty
    if not answer:
        answer = "I'm here to help. Could you share a bit more about what's on your mind?"

    return answer


