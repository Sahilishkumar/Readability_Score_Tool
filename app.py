import html
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from groq import Groq

st.set_page_config(page_title="ChapterLens", page_icon="📖", layout="wide")

# Keep the sentence splitter deliberately lightweight so the app needs no NLP package.
def split_sentences(text):
    return [
        match.group(0)
        for match in re.finditer(r".+?(?:[.!?]+(?:[\"'”’)]*)?(?=\s|$)|$)", text, re.S)
        if match.group(0).strip()
    ]


SYSTEM_PROMPT = """
You are ChapterLens, a careful editorial readability analyst.
Assess the supplied chapter for a general adult reader. Be consistent, evidence-based,
and conservative. Do not judge literary quality, plot, or the author's style.

Return exactly one valid JSON object and nothing else, using this schema:
{
  "readability_score": integer,
  "reading_level": string,
  "summary": string,
  "hardest_sentences": [
    {
      "sentence_index": integer,
      "difficulty": "High" | "Very high",
      "reason": string,
      "suggestion": string
    }
  ]
}

Rules:
- readability_score must be an integer from 0 to 100. Higher means easier to read.
- Use this rubric: 90-100 very easy; 70-89 easy; 50-69 moderate; 30-49 difficult;
  0-29 very difficult. This is an editorial estimate, not a validated formula.
- reading_level should be a short plain-language label, such as "General audience",
  "Advanced adult reader", or "Young adult reader". Do not claim an exact school grade.
- summary must be one concise sentence.
- Select exactly five distinct sentence indices from the numbered sentence list, unless
  fewer than five sentences were supplied. In that case, select every available sentence.
- Sentence indices are 0-based. Copy no sentences into the output; refer to them only by index.
- Rank the selected sentences from hardest to least hard.
- Judge difficulty using syntax, clause nesting, sentence length, uncommon vocabulary,
  ambiguity, and cognitive load. Do not mark a sentence hard merely because its subject
  matter is serious.
- reason must explain the specific difficulty in one short sentence.
- suggestion must offer a concise, faithful simplification without changing the meaning.
- Do not invent facts or assume context that is not in the text.
- Output JSON only. No markdown fences or extra keys.
"""


def build_prompt(sentences):
    numbered = "\n".join(f"{i}: {sentence.strip()}" for i, sentence in enumerate(sentences))
    return f"""Analyze the chapter sentences below.

Return the JSON object required by the system instructions.
There are {len(sentences)} sentences. Use only the provided 0-based indices.

SENTENCES:
{numbered}
"""


def get_api_key():
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def analyze_chapter(sentences, api_key):
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(sentences)},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=1800,
    )
    result = json.loads(response.choices[0].message.content)

    # Validate the model response before displaying it.
    score = result.get("readability_score")
    if not isinstance(score, int) or not 0 <= score <= 100:
        raise ValueError("The model returned an invalid readability score.")

    items = result.get("hardest_sentences")
    if not isinstance(items, list):
        raise ValueError("The model did not return a sentence list.")

    valid_items = []
    seen = set()
    for item in items:
        idx = item.get("sentence_index")
        if isinstance(idx, int) and 0 <= idx < len(sentences) and idx not in seen:
            seen.add(idx)
            valid_items.append(item)

    if len(valid_items) != min(5, len(sentences)):
        raise ValueError("The model returned an incomplete or invalid set of difficult sentences.")

    result["hardest_sentences"] = valid_items
    return result


st.title("📖 ChapterLens")
st.caption("A quick readability check for your chapter, powered by AI.")
st.write("Paste a chapter to get an estimated readability score and the five sentences that may slow readers down.")

chapter = st.text_area(
    "Your chapter",
    height=300,
    placeholder="Paste your chapter here...",
    help="Your text is sent to the Groq API for analysis. Avoid pasting confidential or unpublished material if you are not comfortable sending it to a third-party API.",
)

word_count = len(re.findall(r"\b[\w’'-]+\b", chapter))
st.caption(f"{word_count:,} words")

if st.button("Analyze chapter", type="primary", use_container_width=True):
    if not chapter.strip():
        st.warning("Paste a chapter before analyzing.")
        st.stop()

    sentences = split_sentences(chapter.strip())
    if len(sentences) < 1:
        st.warning("We couldn't identify any sentences. Try adding sentence-ending punctuation.")
        st.stop()

    api_key = get_api_key()
    if not api_key:
        st.error("Missing Groq API key. Add GROQ_API_KEY to your environment or Streamlit secrets.")
        st.stop()

    with st.spinner("Reading your chapter..."):
        try:
            result = analyze_chapter(sentences, api_key)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    score = result["readability_score"]
    col1, col2 = st.columns(2)
    col1.metric("Readability score", f"{score}/100")
    col2.metric("Estimated reading level", result.get("reading_level", "Not available"))
    st.progress(score)
    st.caption("Higher scores indicate easier reading. This score is an AI estimate, not a validated readability formula.")
    st.write(result.get("summary", ""))

    hard_items = result["hardest_sentences"]
    hard_by_index = {item["sentence_index"]: item for item in hard_items}

    st.subheader("Chapter with difficult sentences highlighted")
    highlighted = []
    for i, sentence in enumerate(sentences):
        safe_sentence = html.escape(sentence)
        if i in hard_by_index:
            highlighted.append(
                f'<mark style="background-color:#ffe7a3;padding:2px 3px;border-radius:3px;">{safe_sentence}</mark>'
            )
        else:
            highlighted.append(safe_sentence)

    # HTML-escape user text before inserting it into the small highlighted view.
    st.markdown(
        '<div style="line-height:1.9;font-size:1rem;">' + " ".join(highlighted) + "</div>",
        unsafe_allow_html=True,
    )

    st.subheader("The five hardest sentences")
    for rank, item in enumerate(hard_items, start=1):
        idx = item["sentence_index"]
        with st.container(border=True):
            st.markdown(f"**{rank}. {item.get('difficulty', 'High')} difficulty**")
            st.write(sentences[idx].strip())
            st.caption(f"Why it may be difficult: {item.get('reason', 'No reason supplied.')}")
            st.markdown("**Possible simplification**")
            st.write(item.get("suggestion", "No suggestion supplied."))

    st.caption("AI output can be wrong. Review the highlights and rewrites before making editorial decisions.")
