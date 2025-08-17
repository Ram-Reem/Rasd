
import os
import pandas as pd
import re
import numpy as np
from typing import Dict
from typing import List , Optional


def convert_np_types(obj):
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def is_useless(text) -> bool:
    if pd.isna(text):
        return True

    text = str(text).strip()
    if not text:
        return True

    
    if re.fullmatch(r"[^\w\u0600-\u06FF]+", text):
        return True

    
    if re.fullmatch(r"(آ+|ا{1,4}|هه+|أه+|،،،+|؟؟+)", text):
        return True

    
    if re.fullmatch(r"[A-Za-z]", text):  
        return True
    if re.fullmatch(r"([A-Za-z])\1{2,}", text):  
        return True

    
    if re.fullmatch(r"[A-Za-z0-9\s.,!?;:'\"()\-\[\]{}]+", text):
        words = [w for w in text.split() if len(w) > 2]
        if not words:
            return True  

    return False

def is_empty_or_symbols(text) -> bool:
    if pd.isna(text):
        return True
    text = str(text).strip()
    if not text:
        return True
    return bool(re.fullmatch(r"[^\w\u0600-\u06FF]+", text))


def analyze_data(df: pd.DataFrame, notes_col: str = 'ملاحظات') -> Dict:
    total_rows = len(df)
    empty_mask = df[notes_col].apply(is_empty_or_symbols)
    garbage_mask = df[notes_col].apply(lambda x: is_useless(x) and not is_empty_or_symbols(x))
    result = {
        "total_comments": total_rows,
        "valid_comments": total_rows - (empty_mask.sum() + garbage_mask.sum()),
        "empty_comments": empty_mask.sum(),
        "garbage_comments": garbage_mask.sum()
    }
    return convert_np_types(result)


def export_cleaned_versions(df: pd.DataFrame, notes_col='ملاحظات') -> Dict:
    os.makedirs("outputs", exist_ok=True)
    df_clean = df[~df[notes_col].apply(is_useless)]
    output_path = "outputs/clean_data.xlsx"
    df_clean.to_excel(output_path, index=False)
    return {
        "cleaned_file_path": output_path,
        "cleaned_comments": df_clean[notes_col].astype(str).tolist(),
        "df_clean": df_clean  
    }

