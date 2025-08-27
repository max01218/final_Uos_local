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
		r"^\s*Human:.*?(?=\n|$)",
		r"^\s*(Q:|Question:).*?(?=\n|$)",
		r"^\s*OPTIONS FOR NEXT STEPS:.*?(?=\n|$)",
		r"^\s*Choose the first option:.*?(?=\n|$)",
		r"^\s*Choose (one|an) option:.*?(?=\n|$)",
		r"^\s*Next Steps:.*?(?=\n|$)",
		r"^\s*Assistant asked:.*?(?=\n|$)",
		r"^\s*Assistant:.*?(?=\n|$)",
		r"^\s*Context:.*?(?=\n|$)",
		r"^\s*History:.*?(?=\n|$)",
	]
	for pattern in template_patterns:
		answer = re.sub(pattern, "", answer, flags=re.IGNORECASE | re.MULTILINE)

	# Remove labels like Empathy:, Citation:, Follow-up question: etc (but KEEP "Q:")
	answer = re.sub(
		r"(Empathy:|Citation:|Follow-up question:|User:|Assistant:|Context:|History:|Human:|Question:|Options:|Next Steps:)",
		"",
		answer,
		flags=re.IGNORECASE,
	)

	# Remove ChatML or similar special tokens
	answer = re.sub(r"<\|assistant\|\|?\s*", "", answer, flags=re.IGNORECASE)
	answer = re.sub(r"<\|.*?\|>", "", answer, flags=re.IGNORECASE)
	# Remove truncated special tokens or leaked system tags
	answer = re.sub(r"(?mi)^\s*<\|(system|assistant|user).*?$", "", answer)
	# Remove leading quote markers
	answer = re.sub(r"(?m)^\s*>\s?", "", answer)

	# Remove salutations and repeated name greetings at the beginning
	answer = re.sub(r"(?mi)^\s*(hi|hello|hey|dear)\s+[A-Za-z][A-Za-z\-\s]{0,40}[:,]?\s*", "", answer)
	answer = re.sub(r"(?mi)^\s*(hi|hello|hey|dear)[:,]?\s*", "", answer)

	# Remove filler interjections at line starts (e.g., "Certainly!", "Sure!")
	answer = re.sub(r"(?m)^(Certainly!|Sure!|Absolutely!|Of course!|Great question!|Good question!)[\s,\-]*", "", answer)

	# Remove trailing instruction/output format artifacts
	answer = re.sub(r"(?m)^\s*OUTPUT FORMAT:.*$", "", answer)
	answer = re.sub(r"(?m)^\s*Please respond with .* next question\.?$", "", answer)
	answer = re.sub(r"(?m)^\s*Understood\.?\s*$", "", answer)
	# Remove stray double quotes left by templates
	answer = answer.replace('""', '"').replace("''", "'")

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

	# Remove self-answered rating lines (e.g., "My stress is 4 out of 10.")
	rating_self_reply = r"(?mi)^\s*(My\b.*?out of\s*10\.?|I\b.*?out of\s*10\.?|It\b.*?out of\s*10\.?)\s*$"
	answer = re.sub(rating_self_reply, "", answer)

	# If there is a Q: line, keep only up to and including the final Q: line
	try:
		q_matches = list(re.finditer(r"(?mi)^\s*Q:\s*.*$", answer))
		if q_matches:
			last_q = q_matches[-1]
			# cut everything after the end of the Q line
			answer = answer[: last_q.end()]
			# ensure the final Q line ends with a question mark
			q_text = last_q.group(0)
			if not q_text.rstrip().endswith("?"):
				answer = answer[: last_q.start()] + q_text.rstrip() + "?"
	except Exception:
		pass

	# Finalize dangling follow-up fragments by appending a question mark if needed
	if re.search(r"(would you( like)?|do you want|shall we|are you open to|would it help)\s*$", answer, re.IGNORECASE):
		answer = answer.rstrip() + "?"

	# Ensure the output is not empty
	if not answer:
		answer = "I'm here to help. Could you share a bit more about what's on your mind?"

	return answer

_NAME_SALUTATION = re.compile(r'^\s*(hi|hello|hey)\b[^A-Za-z0-9]*', re.I)
_USER_NAME = re.compile(r'\b(jui\s*chang|jui)\b[:,]?', re.I)
def format_e_s_q_output(raw: str, word_limit: int = 120) -> str:
    if not raw:
        return raw
    e = re.search(r'^\s*E:\s*(.+)$', raw, re.I | re.M)
    s = re.search(r'^\s*S:\s*(.+)$', raw, re.I | re.M)
    q = re.search(r'^\s*Q:\s*(.+)$', raw, re.I | re.M)

    parts = []
    if e: parts.append(e.group(1).strip())
    if s: parts.append(s.group(1).strip())
    if q:
        qtxt = q.group(1).strip()
        if not qtxt.lower().startswith('q:'):
            qtxt = 'Q: ' + qtxt
        parts.append(qtxt)

    # 清除稱呼與名字
    if parts:
        parts[0] = _NAME_SALUTATION.sub('', parts[0])
        parts[0] = _USER_NAME.sub('', parts[0])

    # 去重
    out_lines, seen = [], set()
    for p in parts:
        key = re.sub(r'[^a-z0-9]+', '', p.lower())
        if p and key not in seen:
            seen.add(key)
            out_lines.append(p)
    out = '\n'.join(out_lines[:3]).strip()

    # 字數限制
    words = out.split()
    if len(words) > word_limit:
        out = ' '.join(words[:word_limit])
        out = out.replace(' Q:', '\nQ:')
    return out



