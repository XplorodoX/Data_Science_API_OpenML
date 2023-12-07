import openml

class OpenML_API:
    def __init__(self):
        self.logger = None

    @staticmethod
    def list_datasets(output_format='dataframe'):
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

    # TODO: Filtern nach Anzahl der Features
    def filter_datasets_by_attribute_types(self, start_date=None, end_date=None, num_attributes_range=None,
                                           num_features_range=None, limit=None):
        datasets_list = self.list_datasets()
        dataset_ids = datasets_list['did'].tolist()
        filtered_datasets = []
        not_matching_count = 0

        if start_date and end_date and start_date > end_date:
            raise ValueError("Startdatum muss vor dem Enddatum liegen.")

        for dataset_id in dataset_ids:
            if limit is not None and limit <= 0:
                break

            try:
                dataset = self.get_dataset(dataset_id)
                # dataset_features = dataset.features
                dataset_date = dataset.upload_date

                if (not start_date or start_date < dataset_date) and (not end_date or end_date > dataset_date):

                    filtered_datasets.append((dataset_id, dataset.name))

                    if limit is not None:
                        limit -= 1
                else:
                    not_matching_count += 1

            except Exception as e:
                print(f"Fehler bei der Verarbeitung des Datensatzes {dataset_id}: {e}")

        return filtered_datasets, not_matching_count
