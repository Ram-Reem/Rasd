
import hashlib
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.utils.cleaning import analyze_data, export_cleaned_versions, convert_np_types
from app.utils.sentiment import analyze_sentiment, extract_top_employees
from app.utils.helpers import get_value
from app.utils.file import read_excel_or_csv

router = APIRouter()

cache = {}

def convert_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_types(i) for i in obj]
    return obj


def hash_file_content(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()

@router.post("/upload-clean")
async def upload_and_clean(file: UploadFile = File(...)):
    if not file.filename.endswith((".xlsx", ".csv")):
        raise HTTPException(status_code=400, detail="صيغة الملف غير مدعومة. الرجاء رفع ملف Excel أو CSV.")

    file_bytes = await file.read()
    file_hash = hash_file_content(file_bytes)

    if file_hash in cache:
        return JSONResponse(content=cache[file_hash])

    try:
        df = read_excel_or_csv(file_bytes, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل في قراءة الملف: {str(e)}")

    if "ملاحظات" not in df.columns:
        raise HTTPException(status_code=400, detail="الملف لا يحتوي على عمود 'ملاحظات'.")

    stats = {
        "solved": get_value(df, "هل تم حل المشكلة", "y"),
        "unsolved": get_value(df, "هل تم حل المشكلة", "n"),
        "satisfied": get_value(df, "هل انت راضي عن الحل", "y"),
        "unsatisfied": get_value(df, "هل انت راضي عن الحل", "n"),
    }

    analysis = analyze_data(df)
    exported = export_cleaned_versions(df)
    df_clean = exported["df_clean"]
    comments = exported["cleaned_comments"]
    sentiment = analyze_sentiment(comments)

   
    top_employees = extract_top_employees(df)  
    result = {
    "analysis_summary": analysis,
    **stats,
    "sentiment_counts": sentiment["counts"],
    "classified_comments": sentiment["classified_comments"],
    "cleaned_file_path": exported["cleaned_file_path"],
    "top_employees": top_employees,
}

    
    result = convert_types(result)
    cache[file_hash] = result

    return JSONResponse(content={"message": "تم رفع وتحليل وتصنيف الملف بنجاح.", **result})


