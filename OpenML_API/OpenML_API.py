import openml

class OpenML_API:
    def __init__(self):
        pass
    
    def list_datasets(self, output_format='dataframe'):
        return openml.datasets.list_datasets(output_format=output_format)
    
    def get_dataset(self, dataset_id, download_data=False, download_qualities=False, download_features_meta_data=True):
        return openml.datasets.get_dataset(
            dataset_id,
            download_data=download_data,
            download_qualities=download_qualities,
            download_features_meta_data=download_features_meta_data
        )
    
    def filter_datasets_by_attribute_types(self, attribute_types, limit=None, require_all=True):
        # Stelle sicher, dass attribute_types eine Liste ist
        if not isinstance(attribute_types, list):
            attribute_types = [attribute_types]
        
        # Liste von Datensätzen abrufen
        datasets_list = self.list_datasets()
        
        # Nur Datensatz-IDs extrahieren
        dataset_ids = datasets_list['did'].tolist()
        
        filtered_dataset_names = []
        not_matching_count = 0

        # Über die Datensatz-IDs iterieren
        for dataset_id in dataset_ids:
            dataset = self.get_dataset(dataset_id)
            
            if require_all:
                # Überprüfen, ob der Datensatz alle Attribute der gewünschten Typen enthält
                matches = all(dataset.get_features_by_type(data_type=attr_type) for attr_type in attribute_types)
            else:
                # Überprüfen, ob der Datensatz Attribute eines der gewünschten Typen enthält
                matches = any(dataset.get_features_by_type(data_type=attr_type) for attr_type in attribute_types)
            
            if matches:
                filtered_dataset_names.append(dataset.name)
            else:
                not_matching_count += 1
            
            # Begrenzung überprüfen und Schleife stoppen, wenn das Limit erreicht ist
            if limit and len(filtered_dataset_names) >= limit:
                break

        return filtered_dataset_names, not_matching_count

