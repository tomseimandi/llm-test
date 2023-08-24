from unstructured.partition.pdf import partition_pdf


elements = partition_pdf(
    filename="data/322804147_2020.pdf",
    infer_table_structure=True
)

# print("\n\n".join([str(el) for el in elements]))
tables = [el for el in elements if el.category == "Table"]

print(tables)
