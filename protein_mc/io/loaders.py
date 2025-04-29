"""Light-weight file loaders/parsers."""

import pandas as pd


def get_sequence_from_file(file_path: str, index: int = 0) -> str:
    """
    Read a CSV/TSV/Excel file and return the sequence at *index*.

    The column may be named 'Sequence' or 'sequence'; case-sensitive.
    """
    if file_path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path)
    else:  # CSV / TSV
        sep = "\t" if file_path.lower().endswith(".tsv") else ","
        df = pd.read_csv(file_path, sep=sep)

    for col in ("Sequence", "sequence"):
        if col in df.columns:
            return str(df.iloc[index][col])

    raise ValueError("Input file needs a 'Sequence' column (case sensitive)")
