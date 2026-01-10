import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os
import json

from django.http import JsonResponse
from django.conf import settings
from datetime import datetime
from sklearn.impute import SimpleImputer


def clean_data_view(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': 'Method not allowed'}, status=405)

    try:
        data = json.loads(request.body)
        filename = data.get('filename')
        file_type = data.get('file_type', 'merged')   # merged | output
        action = data.get('action', 'analyze')        # analyze | clean

        if file_type == 'merged':
            file_path = os.path.join(
                settings.BASE_DIR,
                'Weather_Forcast_App', 'Merge_data', filename
            )
        else:
            file_path = os.path.join(
                settings.BASE_DIR,
                'Weather_Forcast_App', 'Output', filename
            )

        if not os.path.exists(file_path):
            return JsonResponse({'success': False, 'message': 'File không tồn tại'})

        data_df = pd.read_excel(file_path)

        if action == 'analyze':
            return JsonResponse({
                'success': True,
                'analysis': analyze_missing_data(data_df, filename)
            })

        if action == 'clean':
            return JsonResponse({
                'success': True,
                **perform_cleaning(data_df, filename)
            })

    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)})
def analyze_missing_data(data_df, filename):
    total_rows = len(data_df)
    total_columns = len(data_df.columns)

    data_df.replace(["N/A", "NA", "null", ""], np.nan, inplace=True)

    missing_report = []
    for col in data_df.columns:
        missing = data_df[col].isna().sum()
        missing_report.append({
            "column": col,
            "missing_count": int(missing),
            "percent": round(missing / total_rows * 100, 2),
            "dtype": str(data_df[col].dtype)
        })

    plt.figure(figsize=(12, 8))
    sns.heatmap(data_df.isna(), cmap="Blues", yticklabels=False)
    plt.title(f"Missing Data Heatmap - {filename}")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    heatmap = base64.b64encode(buf.getvalue()).decode()

    return {
        "filename": filename,
        "total_rows": total_rows,
        "total_columns": total_columns,
        "missing_report": missing_report,
        "heatmap_image": heatmap
    }
def perform_cleaning(data_df, filename):
    original_df = data_df.copy()
    cleaning_log = {}

    # =========================
    # 1. Chuẩn hóa missing
    # =========================
    data_df.replace(["N/A", "NA", "null", ""], np.nan, inplace=True)

    # =========================
    # 2. Xử lý dữ liệu âm
    # =========================
    negative_fixed = {}
    for col in data_df.select_dtypes(include=[np.number]).columns:
        count = (data_df[col] < 0).sum()
        if count > 0:
            data_df.loc[data_df[col] < 0, col] = 0
            negative_fixed[col] = int(count)
    cleaning_log["negative_fixed"] = negative_fixed

    # =========================
    # 3. Chuẩn hóa kiểu dữ liệu
    # =========================
    dtype_log = {}
    for col in data_df.columns:
        if data_df[col].dtype == object:
            converted = pd.to_numeric(data_df[col], errors='ignore')
            if not converted.equals(data_df[col]):
                data_df[col] = converted
                dtype_log[col] = "string → numeric"

        if "date" in col.lower() or "time" in col.lower():
            data_df[col] = pd.to_datetime(data_df[col], errors='coerce')
            dtype_log[col] = "→ datetime"

    cleaning_log["datatype_standardized"] = dtype_log

    # =========================
    # 4. Imputation
    # =========================
    num_cols = data_df.select_dtypes(include=[np.number]).columns
    cat_cols = data_df.select_dtypes(exclude=[np.number]).columns

    if len(num_cols):
        data_df[num_cols] = SimpleImputer(strategy="mean").fit_transform(data_df[num_cols])

    if len(cat_cols):
        data_df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(data_df[cat_cols])

    # =========================
    # 5. Kiểm tra toàn vẹn dữ liệu
    # =========================
    integrity_report = []
    if "category_id" in data_df.columns:
        valid_ids = {1, 2, 3, 4}
        invalid = ~data_df["category_id"].isin(valid_ids)
        if invalid.sum() > 0:
            integrity_report.append({
                "column": "category_id",
                "invalid_count": int(invalid.sum())
            })

    # =========================
    # 6. So sánh trước – sau
    # =========================
    comparison = {
        "rows_before": len(original_df),
        "rows_after": len(data_df),
        "columns_before": original_df.shape[1],
        "columns_after": data_df.shape[1],
        "missing_before": int(original_df.isna().sum().sum()),
        "missing_after": int(data_df.isna().sum().sum())
    }

    # =========================
    # 7. Xuất file CSV
    # =========================
    today = datetime.now().strftime("%d-%m-%Y")
    clean_filename = f"{os.path.splitext(filename)[0]}_cleaned_{today}.csv"

    output_dir = r"D:\Hoc_tap\PROJRCT_WEATHER_FORCAST\PROJRCT_WEATHER_FORCAST\Weather_Forcast_App\Merge_data\clean_data"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, clean_filename)
    data_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # =========================
    # 8. Xuất báo cáo JSON
    # =========================
    report_path = output_path.replace(".csv", "_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "comparison": comparison,
            "cleaning_log": cleaning_log,
            "integrity_report": integrity_report
        }, f, ensure_ascii=False, indent=4)

    return {
        "message": "Làm sạch dữ liệu hoàn tất",
        "output_file": clean_filename,
        "report_file": os.path.basename(report_path)
    }
