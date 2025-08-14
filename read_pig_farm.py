#!/usr/bin/env python3

import sys
from pathlib import Path

try:
	import pandas as pd  # type: ignore
except Exception:  # pylint: disable=broad-except
	pd = None  # type: ignore


def _read_with_pandas(excel_path: Path) -> None:
	excel_file = pd.ExcelFile(excel_path, engine="openpyxl")
	sheet_names = excel_file.sheet_names
	print(f"Found sheets: {sheet_names}")

	first_sheet_name = sheet_names[0]
	dataframe = pd.read_excel(excel_file, sheet_name=first_sheet_name)
	print(f"Loaded sheet: {first_sheet_name} with shape {tuple(dataframe.shape)}")

	print("Columns and dtypes:")
	for column_name, dtype in zip(dataframe.columns, dataframe.dtypes):
		print(f"  - {column_name}: {dtype}")

	print("\nPreview (first 10 rows):")
	print(dataframe.head(10).to_string(index=False))


def _read_with_openpyxl(excel_path: Path) -> None:
	from openpyxl import load_workbook  # type: ignore

	workbook = load_workbook(filename=str(excel_path), read_only=True, data_only=True)
	sheet_names = workbook.sheetnames
	print(f"Found sheets: {sheet_names}")

	first_sheet_name = sheet_names[0]
	sheet = workbook[first_sheet_name]
	print(f"Loaded sheet: {first_sheet_name}")

	rows_iter = sheet.iter_rows(values_only=True)
	try:
		header = next(rows_iter)
	except StopIteration:
		print("Sheet is empty")
		return

	print("Columns:")
	for column_index, column_name in enumerate(header or [], start=1):
		print(f"  - c{column_index}: {column_name}")

	print("\nPreview (first 10 data rows):")
	row_count = 0
	for row in rows_iter:
		print("\t".join("" if cell is None else str(cell) for cell in row))
		row_count += 1
		if row_count >= 10:
			break


def main() -> None:
	"""Load the Excel file, display sheet names, schema, and a small preview."""
	excel_path = Path("/workspace/pig_farm_7_2024.xlsx")
	if not excel_path.exists():
		print(f"Error: file not found: {excel_path}", file=sys.stderr)
		sys.exit(1)

	try:
		if pd is not None:
			_read_with_pandas(excel_path)
		else:
			_read_with_openpyxl(excel_path)
	except Exception as exc:  # pylint: disable=broad-except
		print("Failed to read Excel:", file=sys.stderr)
		print(str(exc), file=sys.stderr)
		sys.exit(2)


if __name__ == "__main__":
	main()