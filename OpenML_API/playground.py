import pandas as pd
import numpy as np

# Beispiel-DataFrame
df = pd.DataFrame({
    'Alter': [25, 30, 35],
    'Geschlecht': ['männlich', 'weiblich', 'divers'],
    'Bildung': ['Hochschule', 'Mittelschule', 'Grundschule'],
    'Noten':['sehr niedrig', 'niedrig', 'mittel'],
    'Nummer':[10, 23, 12]
})

# Identifizieren Sie numerische und kategorische Spalten
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

def is_ordinal(column):
    # Beispiele für bekannte ordinale Skalen
    ordinal_scales_examples = [
        ['Grundschule', 'Mittelschule', 'Hochschule'],
        ['niedrig', 'mittel', 'hoch'],
        ['schlecht', 'ausreichend', 'befriedigend', 'gut', 'sehr gut'],  # Bewertungsskala
        ['gering', 'moderat', 'stark'],  # Intensitätsskala
        ['nie', 'selten', 'manchmal', 'oft', 'immer'],  # Häufigkeitsskala

        # Englische Beispiele
        ['low', 'medium', 'high'],
        ['poor', 'fair', 'good', 'very good', 'excellent'],  # Leistungsbewertung
        ['none', 'mild', 'moderate', 'severe'],  # Schweregrad-Skala
        ['strongly disagree', 'disagree', 'neutral', 'agree', 'strongly agree'],  # Likert-Skala
        ['beginner', 'intermediate', 'advanced', 'expert']  # Kompetenzniveaus
    ]

    unique_values = column.dropna().unique()

    # Überprüfen, ob die Spalte einer bekannten ordinalen Skala entspricht
    for scale in ordinal_scales_examples:
        if set(unique_values) <= set(scale):
            return True
    return False

# Überprüfen der kategorischen Spalten
for col in categorical_cols:
    if is_ordinal(df[col]):
        print(f"Spalte '{col}' ist ordinal")
    else:
        print(f"Spalte '{col}' ist nominal")
