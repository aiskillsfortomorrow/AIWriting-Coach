"""AI Writing Coach

Paste a paragraph -> get grammar corrections, an explanation of each mistake,
and rewrites in simple and formal English. Powered by the Groq API.

Local run:  streamlit run app.py
Deploy:     push to GitHub, then add GROQ_API_KEY in Streamlit Cloud > Secrets
"""

import json
import os
import re

import streamlit as st
from groq import Groq, GroqError

# Default model. If Groq retires it, change this or set GROQ_MODEL in secrets.
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_CHARS = 3000

SYSTEM_PROMPT = """You are a friendly, precise English writing coach.

The user will give you a paragraph. Respond with ONLY a valid JSON object
(no markdown fences, no extra text) using exactly this structure:

{
  "corrected": "the paragraph with grammar, spelling and punctuation fixed, keeping the author's original wording and meaning as much as possible",
  "mistakes": [
    {
      "original": "the exact incorrect phrase from the user's text",
      "correction": "the corrected phrase",
      "category": "Grammar | Spelling | Punctuation | Word choice | Style",
      "explanation": "a short, clear explanation of the rule and why it was wrong"
    }
  ],
  "simple": "a rewrite in plain, easy-to-read English (short sentences, common words)",
  "formal": "a rewrite in polished, formal English suitable for professional or academic use"
}

Rules:
- Keep the original meaning. Do not add new facts.
- If there are no mistakes, return an empty "mistakes" list.
- Explanations must be beginner-friendly."""


def get_secret(name: str) -> str | None:
    """Read a value from Streamlit secrets, falling back to environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


def parse_json(text: str) -> dict:
    """Parse the model's JSON, tolerating accidental code fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def analyze(client: Groq, paragraph: str) -> dict:
    response = client.chat.completions.create(
        model=get_secret("GROQ_MODEL") or DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": paragraph},
        ],
        temperature=0.3,
        max_completion_tokens=4000,
        response_format={"type": "json_object"},
    )
    return parse_json(response.choices[0].message.content)


def main() -> None:
    st.set_page_config(page_title="AI Writing Coach", page_icon="✍️", layout="wide")
    st.title("✍️ AI Writing Coach")
    st.caption("Fix grammar, understand your mistakes, and rewrite in simple or formal English.")

    api_key = get_secret("GROQ_API_KEY")

    with st.sidebar:
        st.header("Settings")
        if not api_key:
            api_key = st.text_input("Groq API key", type="password") or None
        style = st.radio("Rewrite style", ["Both", "Simple", "Formal"], index=0)
        st.markdown("---")
        st.markdown(
            "**How it works**\n\n"
            "1. Paste your paragraph\n"
            "2. Click *Analyze*\n"
            "3. Review corrections, explanations and rewrites"
        )

    paragraph = st.text_area(
        "Your paragraph",
        height=200,
        max_chars=MAX_CHARS,
        placeholder="e.g. Yesterday me and my friend goes to the market for buy some vegetable.",
    )

    if st.button("Analyze", type="primary"):
        if not paragraph.strip():
            st.warning("Please enter a paragraph first.")
            return
        if not api_key:
            st.error("Please add your Groq API key (sidebar, or GROQ_API_KEY in secrets).")
            return

        with st.spinner("Analyzing your writing..."):
            try:
                st.session_state["result"] = analyze(Groq(api_key=api_key), paragraph.strip())
                st.session_state["original"] = paragraph.strip()
            except json.JSONDecodeError:
                st.error("The response couldn't be parsed. Please try again.")
                return
            except GroqError as e:
                st.error(f"API error: {e}")
                return

    result = st.session_state.get("result")
    if not result:
        return

    tab_fix, tab_why, tab_rewrite = st.tabs(["✅ Corrected", "📝 Mistakes explained", "🔄 Rewrites"])

    with tab_fix:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.write(st.session_state.get("original", ""))
        with col2:
            st.subheader("Corrected")
            st.success(result.get("corrected", ""))

    with tab_why:
        mistakes = result.get("mistakes", [])
        if not mistakes:
            st.balloons()
            st.success("No mistakes found. Great job!")
        else:
            st.write(f"Found **{len(mistakes)}** issue(s):")
            for i, m in enumerate(mistakes, 1):
                with st.expander(f"{i}. {m.get('category', 'Issue')}: “{m.get('original', '')}”", expanded=True):
                    st.markdown(f"❌ **Original:** {m.get('original', '')}")
                    st.markdown(f"✅ **Correction:** {m.get('correction', '')}")
                    st.markdown(f"💡 **Why:** {m.get('explanation', '')}")

    with tab_rewrite:
        if style in ("Both", "Simple"):
            st.subheader("Simple English")
            st.info(result.get("simple", ""))
        if style in ("Both", "Formal"):
            st.subheader("Formal English")
            st.info(result.get("formal", ""))

    with st.expander("Copy-friendly text"):
        st.code(result.get("corrected", ""), language=None)


if __name__ == "__main__":
    main()
