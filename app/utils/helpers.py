# Return the count of a specific value in a column after normalization
# (strip, lowercase, convert to string)
def get_value(df, column: str, value: str) -> int | None:
    if column not in df.columns:
        return None
    return df[column].astype(str).str.strip().str.lower().value_counts().get(value, 0)
