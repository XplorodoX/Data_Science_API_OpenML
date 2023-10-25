from OpenML_API import OpenML_API

def main():
    api = OpenML_API()

    dataset_names, not_matching = api.filter_datasets_by_attribute_types(['numeric', 'nominal'], limit=10, require_all=True)
    print(dataset_names)
    print(f"Anzahl der zurückgegebenen Datensätze: {len(dataset_names)}")
    print(f"Anzahl der Datensätze, die nicht beiden Typen entsprechen: {not_matching}")

if __name__ == "__main__":
    main()
