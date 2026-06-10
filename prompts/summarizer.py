def build_summarization_prompt(text: str):

    return f"""
You are an expert summarizer.

Instructions:
1. Summarize in exactly 3 bullet points
2. Each bullet should be under 20 words
3. Use simple English

Content:

###
{text}
###
"""