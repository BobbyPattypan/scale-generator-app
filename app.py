import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product, cycle

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

# Noms connus de gammes (avec ajouts du Thesaurus de Slonimsky)
KNOWN_SCALES = dict({
    (1,2,5,5,2): "Insen (Kokin‑joshi)",
    (1,4,1,5,2): "Iwato",
    (1,2,3,5,1): "Han‑Kumoi (Kumoi‑joshi)",
    (2,3,2,3,2): "Ritsu (Ritsusen pentatonique)",
    (2,1,3,1,3): "Scottish pentatonique",
    (2,2,1,2,4): "Egyptian pentatonique",
    (1,3,1,2,1,2,1): "Spanish Gypsy / Phrygien dominant",
    (1,3,1,1,2,3,1): "Persian (Double Harmonic)",
    (1,2,1,3,1,2,1): "Hungarian Gypsy Minor",
    (2,2,1,2,1,4): "Andine (pentatonique des Andes)",
    (1,3,4,2,2): "Indienne Bhairav (échelle hindoustanie)",
    (2,1,4,1,4): "Indienne Todi (raga Todi pentatonique)",
    (2,2,1,3,1,3): "Celtique (Mixolydien modifié)",
    (3,1,2,3,3): "Africaine équatoriale (intervalles symétriques)",
    (2,2,2,3,3): "Afrique de l’Ouest (Pentatonique avancée)",
    (3,2,2,3,2): "Africaine sahélienne (mode griotique)",
    (2, 1, 2, 2, 2, 2, 1): "Mode Jazz Melodic Minor (I)",
    (1, 1, 2, 2, 2, 2, 2): "Dorian ♭2 (mode II de mélodique mineure)",
    (2, 2, 2, 1, 1, 2, 2): "Lydien augmenté (mode III)",
    (2, 2, 2, 1, 2, 1, 2): "Lydien ♭7 (mode IV, Lydien dominant)",
    (2, 2, 1, 2, 1, 2, 2): "Mixolydien ♭6 (mode V)",
    (1, 2, 2, 1, 1, 2, 2): "Locrian ♯2 (mode VI)",
    (1, 1, 1, 2, 2, 2, 2): "Altéré (mode VII de mélodique mineure)",
    (1, 3, 1, 2, 1, 2, 2): "Phrygien dominant (mode V de la mineure harmonique)",
    (2, 1, 4, 1, 4): "Hirajōshi (pentatonique japonaise)",
    (2, 1, 2, 2, 1, 2, 2): "Mineure harmonique",
    (2, 1, 2, 2, 2, 2, 1): "Mineure mélodique ascendante",
    (1, 2, 2, 1, 2, 2, 2): "Altéré (Super-locrian)",
    (3, 2, 1, 1, 3, 2): "Blues mineure",
    (2, 1, 1, 1, 3, 2, 2): "Blues majeure",
    (2, 2, 1, 2, 2, 1, 1, 2): "Bebop dominante",
    (2, 1, 1, 1, 2, 2, 1, 3): "Bebop dorienne",
    (2, 2, 1, 2, 1, 1, 2, 2): "Bebop majeure",
    (2, 1, 2, 2, 1, 1, 2, 2): "Bebop mélodique mineure",
    (2, 2, 1, 2, 2, 2, 1): "Majeur (Ionien)",
    (2, 1, 2, 2, 2, 1, 2): "Mineur naturel (Éolien)",
    (2, 1, 2, 2, 1, 2, 2): "Dorien",
    (1, 2, 2, 1, 2, 2, 2): "Phrygien",
    (2, 2, 2, 1, 2, 2, 1): "Lydien",
    (2, 2, 1, 2, 2, 1, 2): "Mixolydien",
    (2, 2, 3, 2, 3): "Pentatonique majeure",
    (3, 2, 2, 3, 2): "Pentatonique mineure",
    (1, 2, 1, 2, 2, 4): "Hexatonique blues",
    (2, 1, 1, 1, 2, 2, 3): "Hexatonique lydienne b7",
    (2, 2, 2, 2, 2, 2): "Hexatonique par tons entiers",
    (1, 3, 1, 3, 1, 3): "Hexatonique symétrique (Tierces mineures) [Slonimsky]",
    (3, 3, 3, 3): "Tritonique (Tierces mineures)",
    (4, 4, 4): "Tritonique (Tierces majeures) [Slonimsky]",
    (6, 6): "Bitonale par tritons [Slonimsky]",
    (1, 5, 1, 5): "Cycle de tritons altérés [Slonimsky]",
    (1, 1, 2, 1, 1, 2, 1, 1, 2): "Neuvatonique symétrique [Slonimsky]",
    (1, 2, 1, 2, 1, 2, 1, 2): "Demi-ton / Ton (Octatonique)",
    (2, 1, 2, 1, 2, 1, 2, 1): "Ton / Demi-ton (Octatonique)",
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1): "Chromatique"
})

CATEGORY_LABELS = {
    2: "Deux notes",
    3: "Trois notes",
    4: "Quatre notes",
    5: "Pentatonique",
    6: "Hexatonique",
    7: "Heptatonique",
    8: "Octatonique",
    9: "Neuvatonique",
    12: "Chromatique"
}

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
                    color = 'blue'  # Tonique
                elif degree_index in [3]:
                    color = 'green'  # Sous-dominante
                elif degree_index in [4]:
                    color = 'red'  # Dominante
            ax.plot(x, y, 'o', color=color, markersize=14)
                        if display_mode == "notes":
                label = name
            elif display_mode == "degrés":
                label = degree_map.get(name, name)
            else:
                label = f"{name}
{degree_map.get(name, '')}"
            ax.text(x, y, label, ha='center', va='center', color='white', weight='bold')
        else:
            ax.plot(x, y, 'o', color='lightgray', markersize=10)
            ax.text(x, y, name, ha='center', va='center', color='black')
    return fig

def split_into_tetrachords(intervals):
    mid = len(intervals) // 2
    return intervals[:mid], intervals[mid:]

def generate_sequential_pattern(intervals, repetitions=3):
    return intervals * repetitions

def suggest_progression(chords, tonic='C'):
    if len(chords) < 4:
        return []

    jazz = [chords[1 % len(chords)], chords[4 % len(chords)], chords[0]]  # II - V - I
    pop = [chords[0], chords[3 % len(chords)], chords[4 % len(chords)], chords[5 % len(chords)]]  # I - IV - V - vi
    blues = [chords[0], chords[3 % len(chords)], chords[0], chords[0],
             chords[3 % len(chords)], chords[3 % len(chords)], chords[0], chords[0],
             chords[4 % len(chords)], chords[3 % len(chords)], chords[0], chords[4 % len(chords)]]

        return {"Jazz II-V-I": jazz, "Pop I-IV-V-vi": pop, "Blues 12-bar": blues}
    # Exemple simple : I - IV - V - I
    root_names = [root for root, _ in chords]
    return [
        chords[0],  # I
        chords[3 % len(chords)],  # IV
        chords[4 % len(chords)],  # V
        chords[0]   # I
    ]

from fpdf import FPDF

# --- Interface Streamlit ---
st.title("🎶 Générateur de Gammes Musicales Personnalisées")

category_filter = st.selectbox("Filtrer par catégorie", ["Toutes", "🎷 Jazz/Blues", "🌍 Orientales", "🎼 Occidentales", "🧪 Expérimental / Slonimsky", "🌐 Traditionnelles / Monde"] + list(set(CATEGORY_LABELS.values())))
tonic = st.selectbox("Choisir la tonique", NOTE_NAMES)

gamme_pool = []
for n in range(3, 13):
    gamme_pool.extend(generate_valid_scales(n))
if category_filter != "Toutes":
    if category_filter in CATEGORY_LABELS.values():
        target_counts = [k for k, v in CATEGORY_LABELS.items() if v == category_filter]
        gamme_pool = [g for g in gamme_pool if len(g) in target_counts]
    elif category_filter in ["Jazz/Blues", "🎷 Jazz/Blues"]:
        jazz_keywords = ["bebop", "blues", "altéré", "dorien", "lydien", "mixolydien"]
        gamme_pool = [g for g in gamme_pool if KNOWN_SCALES.get(g, "").lower() and any(k in KNOWN_SCALES.get(g, "").lower() for k in jazz_keywords)]
    elif category_filter in ["Orientales", "🌍 Orientales"]:
        oriental_keywords = ["maqam", "orient", "hindou", "byzantin"]
        gamme_pool = [g for g in gamme_pool if KNOWN_SCALES.get(g, "").lower() and any(k in KNOWN_SCALES.get(g, "").lower() for k in oriental_keywords)]
    elif category_filter in ["Occidentales", "🎼 Occidentales"]:
        western_keywords = ["majeur", "mineur", "ionien", "dorien", "phrygien", "lydien", "mixolydien", "éolien", "locrien"]
        gamme_pool = [g for g in gamme_pool if KNOWN_SCALES.get(g, "").lower() and any(k in KNOWN_SCALES.get(g, "").lower() for k in western_keywords)]
    elif category_filter in ["Expérimental / Slonimsky", "🧪 Expérimental / Slonimsky"]:
        gamme_pool = [g for g in gamme_pool if "[slonimsky]" in KNOWN_SCALES.get(g, "").lower()]
elif category_filter in ["Traditionnelles / Monde", "🌐 Traditionnelles / Monde"]:
        keywords = ["afrique", "indienne", "andine", "celtique", "japonaise"]
        gamme_pool = [g for g in gamme_pool if KNOWN_SCALES.get(g, "").lower() and any(k in KNOWN_SCALES.get(g, "").lower() for k in keywords)]

search_term = st.text_input("🔎 Rechercher une gamme par nom (optionnel)").lower()

if search_term:
    gamme_pool = [g for g in gamme_pool if search_term in KNOWN_SCALES.get(g, "").lower()]

scale_options = [(KNOWN_SCALES.get(s, str(s)), s) for s in gamme_pool]

if search_term:
    scale_options = [(name, val) for name, val in scale_options if search_term in name.lower()]
selected = st.selectbox("Choisir une gamme générée", scale_options, format_func=lambda x: x[0])

selected_intervals = selected[1]
real_notes = intervals_to_notes(selected_intervals, tonic)

st.markdown(f"**Tonique** : {tonic}")
st.markdown(f"**Intervalles** : {selected_intervals}")
st.markdown(f"**Notes et degrés** : {' - '.join(f'{note} ({i+1})' for i, note in enumerate(real_notes))}")

display_mode = st.radio("Afficher dans le cercle chromatique :", ["notes", "degrés", "les deux"], index=2)
highlight = st.checkbox("Colorier selon fonction tonale", value=True)
fig = plot_scale_circle(real_notes, display_mode=display_mode, highlight_functions=highlight)
st.pyplot(fig)

if len(selected_intervals) >= 4:
    tet1, tet2 = split_into_tetrachords(selected_intervals)
    st.markdown(f"**Tétracorde 1** : {tet1}")
    st.markdown(f"**Tétracorde 2** : {tet2}")

if st.checkbox("Générer motif séquentiel (style Schoenberg)"):
    repetitions = st.slider("Nombre de répétitions", 2, 8, 3)
    seq = generate_sequential_pattern(selected_intervals, repetitions)
    seq_notes = intervals_to_notes(seq, tonic)
    st.markdown(f"**Motif séquentiel** ({repetitions}x) : {seq}")
    st.markdown(f"**Notes résultantes** : {' - '.join(seq_notes)}")

if st.checkbox("Afficher l'harmonisation de la gamme"):
    seventh = st.checkbox("Inclure la septième (accords à 4 sons)")
    chords = harmonize_scale(real_notes, with_seventh=seventh)
    for root, chord in chords:
        symbol = root
        interval_map = [(NOTE_NAMES.index(n) - NOTE_NAMES.index(root)) % 12 for n in chord[1:]]
        symbol += {
            (4, 7): "",
            (3, 7): "m",
            (3, 6): "dim",
            (4, 8): "aug",
            (4, 7, 10): "7",
            (3, 7, 10): "m7",
            (3, 6, 10): "m7b5",
            (3, 6, 9): "dim7",
            (4, 7, 11): "maj7",
            (3, 7, 11): "m(maj7)"
        }.get(tuple(interval_map), "")
        st.markdown(f"**{symbol}** : {' - '.join(chord)}")

    if st.checkbox("🎼 Suggérer une grille harmonique"):
        options = suggest_progression(chords, tonic)
        progression_name = st.selectbox("Type de progression", list(options.keys()))
        progression = options[progression_name]
        st.markdown(f"**{progression_name}**")
        grid_lines = []
        st.markdown("**Copie texte de la grille :**")
        text_block = "
".join(grid_lines)
        if text_block:
            st.text_area("Grille harmonique (texte)", value=text_block, height=150)
        for i, (root, chord) in enumerate(progression):
            try:
                degree_index = real_notes.index(root)
                deg_symbol_list = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
                function_names = ["Tonique", "Sus-tonique", "Médiante", "Sous-dominante", "Dominante", "Sus-dominante", "Sensible"]
                deg_symbol = deg_symbol_list[degree_index]
                deg_name = function_names[degree_index] if degree_index < len(function_names) else ""
                label = f"{deg_symbol} ({deg_name})"
            except ValueError:
                label = "?"
            symbol = root
            interval_map = [(NOTE_NAMES.index(n) - NOTE_NAMES.index(root)) % 12 for n in chord[1:]]
            symbol += {
                (4, 7): "",
                (3, 7): "m",
                (3, 6): "dim",
                (4, 8): "aug",
                (4, 7, 10): "7",
                (3, 7, 10): "m7",
                (3, 6, 10): "m7b5",
                (3, 6, 9): "dim7",
                (4, 7, 11): "maj7",
                (3, 7, 11): "m(maj7)"
            }.get(tuple(interval_map), "")
            line = f"{label} : {symbol} → {' - '.join(chord)}"
            st.markdown(f"- {line}")
            grid_lines.append(line)

        include_all_chords = st.checkbox("Inclure l'harmonisation complète dans le PDF")
include_notes = st.checkbox("Ajouter des commentaires pédagogiques", value=True)

        if st.button("📄 Exporter la grille en PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt=f"Grille harmonique : {progression_name}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Tonique : {tonic}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Intervalles : {selected_intervals}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Gamme : {', '.join(real_notes)}", ln=True, align='L')
            pdf.cell(200, 10, txt="", ln=True)

            # Visualisation cercle chromatique
            try:
                from io import BytesIO
                img_buf = BytesIO()
                fig = plot_scale_circle(real_notes, display_mode=display_mode, highlight_functions=highlight)
                fig.savefig(img_buf, format="PNG")
                img_buf.seek(0)
                pdf.image(img_buf, x=10, y=pdf.get_y(), w=100)
                pdf.ln(60)
            except Exception as e:
                pdf.cell(200, 10, txt="[Erreur affichage cercle]", ln=True)


            pdf.cell(200, 10, txt="", ln=True)
            for line in grid_lines:
                pdf.cell(200, 10, txt=line, ln=True, align='L')
                        if include_all_chords:
                pdf.cell(200, 10, txt="", ln=True)
                pdf.cell(200, 10, txt="Harmonisation complète :", ln=True, align='L')
                for root, chord in chords:
                    symbol = root
                    interval_map = [(NOTE_NAMES.index(n) - NOTE_NAMES.index(root)) % 12 for n in chord[1:]]
                    symbol += {
                        (4, 7): "",
                        (3, 7): "m",
                        (3, 6): "dim",
                        (4, 8): "aug",
                        (4, 7, 10): "7",
                        (3, 7, 10): "m7",
                        (3, 6, 10): "m7b5",
                        (3, 6, 9): "dim7",
                        (4, 7, 11): "maj7",
                        (3, 7, 11): "m(maj7)"
                    }.get(tuple(interval_map), "")
                    pdf.cell(200, 10, txt=f"{symbol} : {' - '.join(chord)}", ln=True, align='L')

            if include_notes:
                pdf.add_page()
                pdf.cell(200, 10, txt="Commentaires pédagogiques", ln=True, align='L')
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, txt="Cette grille harmonique suit une progression classique du style sélectionné. Les fonctions tonales (Tonique, Sous-dominante, Dominante, etc.) sont identifiées pour faciliter l’analyse harmonique. Les accords sont dérivés directement de la gamme sélectionnée en respectant les règles d’harmonisation. Vous pouvez utiliser cette grille comme base pour la composition ou l’improvisation.")

            pdf_output = "grille_harmonique.pdf"
            pdf.output(pdf_output)
            with open(pdf_output, "rb") as f:
                st.download_button("📥 Télécharger le PDF", f, file_name=pdf_output, mime="application/pdf")

if st.button("📥 Télécharger toutes les gammes connues en CSV"):
    df = pd.DataFrame([
        {"Nom": name, "Intervalles": str(intervals), "Nombre de notes": len(intervals)}
        for intervals, name in KNOWN_SCALES.items()
    ])
    st.download_button("Télécharger CSV", data=df.to_csv(index=False), file_name="gammes_connues.csv", mime="text/csv")

st.info("Affichage de la portée désactivé pour compatibilité avec le déploiement web. Utilisez une version locale pour voir la portée.")
