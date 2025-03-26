import pytest
import streamlit as st

def test_streamlit_load():
    try:
        st.write("Test")
    except Exception:
        pytest.fail("Streamlit ne se charge pas correctement !")
