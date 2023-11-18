import openml

class OpenML_API:
    def __init__(self):
        pass
    
    def list_datasets(self, output_format='dataframe'):
        return openml.datasets.list_datasets(output_format=output_format)
    
    def get_dataset(self, dataset_id, download_data=False, download_qualities=False, download_features_meta_data=True):
        try:
            return openml.datasets.get_dataset(
                dataset_id,
                download_data=download_data,
                download_qualities=download_qualities,
                download_features_meta_data=download_features_meta_data
            )
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des Datensatzes {dataset_id}: {e}")
            raise

    def donwload_dataset(self, dataset_id):
        try:
            return openml.datasets.get_dataset(dataset_id, download_data=True)
        except Exception as e:
            self.logger.error(f"Fehler beim Herunterladen des Datensatzes {dataset_id}: {e}")
            raise
    
    def filter_datasets_by_attribute_types(self, attribute_types, start_date=None, end_date=None, num_attributes_range=None, num_features_range=None, limit=None, require_all=True):
        if not isinstance(attribute_types, list):
            attribute_types = [attribute_types]
        
        datasets_list = self.list_datasets()
        dataset_ids = datasets_list['did'].tolist()
        filtered_dataset_names = []
        not_matching_count = 0

        for dataset_id in dataset_ids:
            dataset = self.get_dataset(dataset_id)
            try:
                # Überprüfen der Attributtypen
                if require_all:
                    matches = all(dataset.get_features_by_type(data_type=attr_type) for attr_type in attribute_types)
                else:
                    matches = any(dataset.get_features_by_type(data_type=attr_type) for attr_type in attribute_types)

                # Zusätzliche Filterlogik hier einfügen
                # z.B. Überprüfung des Datumsbereichs, der Anzahl der Datenpunkte und der Anzahl der Features

                if matches:
                    features = dataset.features
                    # Weitere Informationen aus dem Datensatz extrahieren, wie z.B. das Erstellungsdatum
                    filtered_dataset_names.append((dataset.name, features))
            except Exception as e:
                print(f"Fehler beim Verarbeiten des Datensatzes {dataset_id}: {e}")
                matches = False

            if limit and len(filtered_dataset_names) >= limit:
                break

        return filtered_dataset_names, not_matching_count