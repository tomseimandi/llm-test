from unstructured.partition.pdf import partition_pdf


elements = partition_pdf(
    filename="../document_qa_deploy/orange/379984891_2021.pdf",
    ocr_languages="fra",
    strategy="ocr_only",
    infer_table_structure=False
)

print("\n\n".join([str(el) for el in elements]))
# tables = [el for el in elements if el.category == "Table"]

# print(tables)
