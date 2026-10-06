"""Streamlit demo.  Run from the repo root:  streamlit run app/app.py"""
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.inference import EduSwitchPredictor

COLORS = {"EN": "#cfe8ff", "UR": "#ffe0b3", "NUM": "#e2d6ff", "OTHER": "#eeeeee"}
EXAMPLES = [
    "Sir meri attendance 72% hai can I still sit in the final?",
    "Mujhe assignment submit karni hai before Friday",
    "Mera LMS login nahi ho raha",
    "Can I withdraw this course?",
    "Fee challan kab generate hoga?",
]

st.set_page_config(page_title="EduSwitch", page_icon="🗣️")
st.title("EduSwitch")
st.caption("Urdu and English code switched student query understanding")


@st.cache_resource
def load():
    return EduSwitchPredictor()


pred = load()
example = st.selectbox("Try an example", [""] + EXAMPLES)
text = st.text_area("Student query", value=example, height=90)

if st.button("Analyze", type="primary") and text.strip():
    r = pred.predict(text)
    c1, c2, c3 = st.columns(3)
    c1.metric("Intent", r["intent"].replace("_", " "))
    c2.metric("Confidence", f"{r['confidence'] * 100:.1f}%")
    c3.metric("Code switched", "Yes" if r["code_switched"] else "No")
    st.markdown("**Token analysis**")
    html = " ".join(
        f'<span style="background:{COLORS.get(t["lang"], "#eee")};padding:2px 6px;border-radius:4px;color:#111">'
        f'{escape(t["token"])} <small>{t["lang"]}</small></span>' for t in r["tokens"])
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"Backends: intent={r['backend']['intent']}, language ID={r['backend']['lid']}")
