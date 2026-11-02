import gradio as gr

from app import studio


def test_build(engine, library):
    assert isinstance(studio.build(engine, library), gr.Blocks)
