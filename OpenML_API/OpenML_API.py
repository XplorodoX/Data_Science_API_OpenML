import logging
import openml

class OpenML_API:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def list_datasets(output_format='dataframe'):
        return openml.datasets.list_datasets(output_format=output_format)

    def get_dataset(self, dataset_id, download_data=True, download_qualities=True, download_features_meta_data=True):
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

    def dimension(self, dataset_id):
        """
        Retrieves the dimensions (number of features and instances) of a given dataset.

        :param dataset_id: The ID of the dataset.
        :return: A tuple (num_features, num_instances) representing the dataset's dimensions.
        """
        try:
            # Fetch only dataset qualities to avoid downloading the data if not necessary
            dataset = openml.datasets.get_dataset(dataset_id, download_data=False)

            # Check if dimensions are available in dataset qualities
            if dataset.qualities:
                num_instances = dataset.qualities.get('NumberOfInstances')
                num_features = dataset.qualities.get('NumberOfFeatures')

                if num_instances is not None and num_features is not None:
                    return num_features, num_instances

            # If qualities are not available or don't contain dimensions, download data
            data, _, _, _ = dataset.get_data()
            num_features = data.shape[1]
            num_instances = data.shape[0]
            return num_features, num_instances

        except Exception as e:
            self.logger.error(f"Error retrieving dimensions of dataset {dataset_id}: {e}")
            raise

    def download_dataset(self, dataset_id, preferred_format='parquet'):
        try:
            dataset = openml.datasets.get_dataset(dataset_id, download_data=True)

            # Überprüfen, ob das bevorzugte Format verfügbar ist
            if preferred_format in dataset.format:
                # Logik zum Herunterladen im bevorzugten Format
                pass
            else:
                # Fallback auf ein anderes Format, z.B. ARFF
                pass

        except Exception as e:
            self.logger.error(
                f"Fehler beim Herunterladen des Datensatzes {dataset_id} im Format {preferred_format}: {e}")
            # Implementieren Sie hier einen Fallback-Mechanismus oder geben Sie eine Fehlermeldung aus
            raise

    def filter_datasets_by_attribute_types(self, start_date=None, end_date=None, num_attributes_range=None,
                                           num_features_range=None, limit=None):
        """
                Filters datasets based on upload dates and number of features.

                :param start_date: Minimum upload date for the datasets.
                :param end_date: Maximum upload date for the datasets.
                :param num_features_range: Tuple or list with two elements specifying the range of number of features.
                :param limit: Maximum number of datasets to return.
                :return: A list of filtered datasets.
                """

        datasets_list = self.list_datasets()
        dataset_ids = datasets_list['did'].tolist()
        filtered_datasets = []

        if start_date and end_date and start_date > end_date:
            raise ValueError("Startdatum muss vor dem Enddatum liegen.")

        for dataset_id in dataset_ids:
            if limit is not None and limit <= 0:
                break

            try:
                dataset = self.get_dataset(dataset_id)
                dataset_date = dataset.upload_date
                num_columns, num_rows = self.dimension(dataset_id)

                if ((not start_date or start_date <= dataset_date) and
                        (not end_date or end_date >= dataset_date) and
                        (not num_features_range or num_features_range[0] <= num_columns <= num_features_range[1])):

                    filtered_datasets.append((dataset_id, dataset.name, num_rows, num_columns))

                    if limit is not None:
                        limit -= 1

            except Exception as e:
                print(f"Fehler bei der Verarbeitung des Datensatzes {dataset_id}: {e}")

        return filtered_datasets
