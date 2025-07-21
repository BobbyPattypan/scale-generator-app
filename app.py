import ast

# ...

# Sélection de gamme connue
selected_scale = st.selectbox(
    "Choisir une gamme :",
    [f"{name} ({tuple(scale)})" for scale, name in filtered_scales.items()]
)

# Intervalles extraits de manière sûre
scale_str = selected_scale.split("(")[-1].strip(")")
scale_intervals = ast.literal_eval(scale_str)

# ...

# --- Export PDF ---
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 14)
        self.cell(0, 10, "Gamme musicale", ln=True, align='C')

    def chapter_body(self, name, tonic, notes, chords, chords7):
        self.set_font("Arial", '', 12)
        self.ln(5)
        self.multi_cell(0, 10, f"Nom : {name}\nTonique : {tonic}\nNotes : {', '.join(notes)}\n\nAccords triadiques :\n" + '\n'.join(chords) + "\n\nAccords de septième :\n" + '\n'.join(chords7))

    def print_chapter(self, name, tonic, notes, chords, chords7):
        self.add_page()
        self.chapter_body(name, tonic, notes, chords, chords7)

if st.button("📄 Exporter en PDF"):
    pdf = PDF()
    scale_name = KNOWN_SCALES.get(scale_intervals, "Gamme personnalisée")
    pdf.print_chapter(scale_name, tonic, notes_in_scale, triads, sevenths)
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    st.download_button("Télécharger le PDF", pdf_output.read(), file_name="gamme.pdf")
