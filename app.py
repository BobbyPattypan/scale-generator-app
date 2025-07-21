# app.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product, cycle
from fpdf import FPDF
from io import BytesIO

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

# KNOWN_SCALES et CATEGORY_LABELS (inchangés pour concision ici)
# ... insérer ici les définitions connues de gammes et catégories ...

@st.cache_data
def generate_valid_scales(n_notes):
    valid_scales = set()
    def recurse(current, total):
        if len(current) == n_notes:
            if total == 12:
                valid_scales.add(tuple(current))
            return
        for step in range(1, 7):
            if total + step <= 12:
                recurse(current + [step], total + step)
    recurse([], 0)
    def normalize(scale):
        return min(tuple(scale[i:] + scale[:i]) for i in range(len(scale)))
    return sorted(set(normalize(list(s)) for s in valid_scales))

def intervals_to_notes(intervals, tonic='C'):
    tonic_index = NOTE_NAMES.index(tonic)
    notes = [tonic_index]
    current = tonic_index
    for step in intervals:
        current = (current + step) % 12
        notes.append(current)
    return [NOTE_NAMES[i] for i in notes[:-1]]

def harmonize_scale(notes, with_seventh=False):
    harmonies = []
    semitone_map = {note: i for i, note in enumerate(NOTE_NAMES)}
    note_indices = [semitone_map[n] for n in notes]
    for i, root in enumerate(note_indices):
        third = (root + 4) % 12 if (root + 4) % 12 in note_indices else (root + 3) % 12
        fifth = (root + 7) % 12 if (root + 7) % 12 in note_indices else (root + 6) % 12
        chord = [NOTE_NAMES[root], NOTE_NAMES[third], NOTE_NAMES[fifth]]
        if with_seventh:
            seventh = (root + 10) % 12 if (root + 10) % 12 in note_indices else (root + 11) % 12
            chord.append(NOTE_NAMES[seventh])
        harmonies.append((NOTE_NAMES[root], chord))
    return harmonies

def plot_scale_circle(scale_notes, display_mode="both", highlight_functions=True):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    degree_map = {note: f"{roman}" for roman, note in zip(
        ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"],
        scale_notes)
    }
    for i, name in enumerate(NOTE_NAMES):
        angle = 2 * np.pi * i / 12
        x = np.cos(angle)
        y = np.sin(angle)
        if name in scale_notes:
            degree_index = scale_notes.index(name)
            color = 'blue'
            if highlight_functions:
                if degree_index == 0:
                    color = 'blue'
                elif degree_index == 3:
                    color = 'green'
                elif degree_index == 4:
                    color = 'red'
            ax.plot(x, y, 'o', color=color, markersize=14)
            if display_mode == "notes":
                label = name
            elif display_mode == "degrés":
                label = degree_map.get(name, name)
            else:
                label = f"{name}\n{degree_map.get(name, '')}"
            ax.text(x, y, label, ha='center', va='center', color='white', weight='bold')
        else:
            ax.plot(x, y, 'o', color='lightgray', markersize=10)
            ax.text(x, y, name, ha='center', va='center', color='black')
    return fig

# --- Interface Streamlit simplifiée ---
st.title("🎶 Générateur de Gammes Musicales Personnalisées")
tonics = NOTE_NAMES
selected_tonic = st.selectbox("Choisir la tonique", tonics)
notes_count = st.slider("Nombre de notes dans la gamme", 3, 12, 7)
scales = generate_valid_scales(notes_count)
selected_scale = st.selectbox("Choisir une structure d'intervalles", scales)
real_notes = intervals_to_notes(selected_scale, selected_tonic)

st.markdown(f"**Tonique** : {selected_tonic}")
st.markdown(f"**Intervalles** : {selected_scale}")
st.markdown(f"**Notes** : {' - '.join(real_notes)}")

if st.checkbox("Afficher le cercle chromatique"):
    display_mode = st.radio("Afficher :", ["notes", "degrés", "les deux"], index=2)
    fig = plot_scale_circle(real_notes, display_mode)
    st.pyplot(fig)

if st.checkbox("Afficher l'harmonisation"):
    seventh = st.checkbox("Inclure la septième", value=False)
    chords = harmonize_scale(real_notes, with_seventh=seventh)
    for root, chord in chords:
        st.markdown(f"**{root}** : {' - '.join(chord)}")
