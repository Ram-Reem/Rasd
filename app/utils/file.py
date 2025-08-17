import pandas as pd
from io import BytesIO

# Read a CSV or Excel file from bytes and return it as a pandas DataFrame
def read_excel_or_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    try:
        if filename.endswith(".csv"):
            return pd.read_csv(BytesIO(file_bytes))
        elif filename.endswith((".xlsx", ".xls")):
            return pd.read_excel(BytesIO(file_bytes), engine="openpyxl")
        else:
            raise ValueError("صيغة الملف غير مدعومة.")
    except Exception as e:
        raise RuntimeError(f"فشل في قراءة الملف: {str(e)}")
