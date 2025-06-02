"""
demo.py – load the MINDS‑14 dataset, inspect it,
play random audio samples in a small Gradio app,
and plot one waveform.
"""

from datasets import load_dataset
import gradio as gr
import matplotlib.pyplot as plt
import librosa.display
import random

# ------ 1. Load the Australian‑English subset of MINDS‑14
minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
print(minds)                     # sanity‑check
example = minds[0]               # just one row, nothing special
print("\nFirst example:\n", example)

# convenience: convert label IDs → label strings
id2label = minds.features["intent_class"].int2str

# ------ 2. Optional column cleanup (exactly like tutorial)
columns_to_drop = ["lang_id", "english_transcription"]
minds = minds.remove_columns(columns_to_drop)

# ------ 3. Make a small Gradio app that plays 4 random clips
def random_clip():
    ex = minds[random.randrange(len(minds))]
    audio = ex["audio"]
    label = id2label(ex["intent_class"])
    return (audio["sampling_rate"], audio["array"]), label

with gr.Blocks() as demo:
    gr.Markdown("## MINDS‑14 – listen to four random Australian‑English queries")
    for _ in range(4):
        audio, label = random_clip()
        gr.Audio(value=audio, label=label)

# ------ 4. Plot the waveform of *example* (tutorial’s final step)
array = example["audio"]["array"]
sr = example["audio"]["sampling_rate"]
plt.figure(figsize=(10, 3))
librosa.display.waveshow(array, sr=sr)
plt.title(f"Waveform – intent: {id2label(example['intent_class'])}")
plt.tight_layout()
plt.show()

# ------ 5. Launch the app last (so the plot shows first)
demo.launch()

