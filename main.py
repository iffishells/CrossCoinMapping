from App.utils.helper_similarity_metrics import calculate_dtw_distance, calculate_error_metrics, smape_distance_metric
from App.utils.helper_visualization_functions import plot_and_save, cluster_visualization_of_time_series
from App.utils.helper_visualization_functions import generate_wordclouds_for_clusters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import csv
import glob
import tqdm
import gc
from dotenv import dotenv_values
import logging
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from App.adoptors.pinecone_client import PineconeStorage
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import dotenv_values
config = dotenv_values(os.path.join('.env'))
import requests

# Configure logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - Line: %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler("app_logs.log"),  # Specify your desired .log file name here
        logging.StreamHandler()  # This keeps logging output to the console as well
    ]
)
logger = logging.getLogger(__name__)

embedding_root_path = os.path.join("Datasets","ProcessedData","Embeddings")
os.makedirs(embedding_root_path,exist_ok=True)

class CrossMapping:
    def __init__(self):
        self.directory_names = {
            "dataset_dir_name": "Datasets",
            "preprocessed_data_dir_name": os.path.join("Datasets", "ProcessedData"),
            "visualization_data_dir_name": "ResultsVisualizationDataOfSameToken",
            "testing_garbage_dir_name": "TestingGarbage",
            "ResultsDirectory": "SimilarityResults",
            "cluster_dir_path": "ClusterResultsVisualization",
            "cluster_token_name_dir": "ClusterBaseOnTokenNameVisualization",
            "embedding_root_path": os.path.join("Datasets","ProcessedData","Embeddings")
        }
        for key, value in self.directory_names.items():
            logger.info(f"Creating Directories : {value}")
            os.makedirs(self.directory_names[key], exist_ok=True)

        self.files_path = {
            'raw_price_data': os.path.join("Datasets", "raw_datasets", "prices.csv"),
            'raw_token_names': os.path.join("Datasets", "raw_datasets", "token_names.csv"),
            'raw_token_names_with_embeddings': os.path.join("Datasets", "ProcessedData",
                                                            "token_names_with_embedding.csv"),
            "token_names": os.path.join("TestingGarbage", "token_names.csv"),
            "similarity_results_file_path": os.path.join("Datasets", "ProcessedData",
                                                         "similarity_results_version.csv"),
            "price_level_clustered_result_file_path": os.path.join('Datasets', 'ProcessedData',
                                                                   'cluster_results_time_series.csv')
        }

    def read_price_data(self, path=None):
        cols_to_ignore = ['Unnamed: 0']
        raw_price_df = pd.read_csv(
            path,
            compression='gzip',
            usecols=lambda col: col not in cols_to_ignore
        )
        return raw_price_df



    def add_embedding_if_missing(self, row):
        # Check if embedding already exists
        if pd.notnull(row.get('embedding')) and row['embedding'] is not None:
            return row['embedding']

        # Fetch the embedding for display_name if not present
        embedding = self.get_embeddings([row['display_name']])
        return embedding[0] if embedding else None

    def read_token_names_data(self, path=None):

        if os.path.exists(path):
            token_names_df = pd.read_csv(path)
            return token_names_df

    def filter_price_data(self, price_data=None, price_points_threshold=1400):

        currency_count = price_data['base_currency'].value_counts()
        currency_index_less_then_threshold = currency_count[currency_count > price_points_threshold].index
        price_data_dataframe = price_data[price_data['base_currency'].isin(currency_index_less_then_threshold)]

        same_min_max = price_data_dataframe.groupby('base_currency')['open'].agg(['min', 'max'])
        same_min_max_equal = same_min_max[same_min_max['min'] == same_min_max['max']]

        price_data_dataframe = price_data_dataframe[
            ~price_data_dataframe['base_currency'].isin(same_min_max_equal.index)]
        return price_data_dataframe

    def filter_token_names(self, price_data=None, token_data=None):

        total_tokens, total_tokens_list = len(list(price_data['base_currency'].unique())), list(
            price_data['base_currency'].unique())
        token_names_df = token_data[token_data['id'].isin(total_tokens_list)]

        return token_names_df

    def process_embedding_batch(self, display_names):
        try:
            embeddings = self.get_embeddings(display_names)
            return embeddings
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return [None] * len(display_names)

    def process_embedding(self, index, display_name):
        try:
            # Get the embedding
            response = self.get_embeddings([display_name])

            # Check if the response is not empty and get the embedding
            if response and isinstance(response[0], dict) and 'embedding' in response[0]:
                embedding = response[0]['embedding']

                # Ensure the embedding is the correct shape
                if isinstance(embedding, (list, np.ndarray)):  # or whatever type you're expecting
                    return index, embedding  # Return index and embedding
            else:
                print(f"No valid embedding found for '{display_name}'")
                return index, None

        except Exception as e:
            print(f"Error processing '{display_name}': {e}")
            return index, None

    def process_pair(self,
                     token1,
                     token2,
                     price_pivot_df,
                     directory_names,
                     results_file,
                     smape_threshold):
        logger.info(f"Processing Token1 : {token1} & Token2 : {token2}")
        try:
            token_folder_name = os.path.join(directory_names['visualization_data_dir_name'])

            plot_filename = f"{str(token1)}_vs_{str(token2)}_dtw.png"
            saving_file_path = os.path.join(token_folder_name, plot_filename)

            # Check if the plot already exists
            if os.path.exists(saving_file_path):
                logger.info("Already exist ")
                return

            # Drop missing values and find common indices
            ts1 = price_pivot_df[token1].dropna()
            ts2 = price_pivot_df[token2].dropna()

            common_index = ts1.index.intersection(ts2.index)

            ts1_common = ts1.loc[common_index]
            ts2_common = ts2.loc[common_index]

            # Proceed if common data points exist
            if len(ts1_common) > 0 and len(ts2_common) > 0:
                # Calculate similarity metrics
                dtw_distance = calculate_dtw_distance(ts1_common, ts2_common)
                # cosine_sim = calculate_cosine_similarity_char(token1, token2)

                # Calculate error metrics
                mae, rmse, mape, smape = calculate_error_metrics(ts1_common, ts2_common)
                similarity_score = {
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                    "smape": smape,
                    "dtw": dtw_distance
                    # "cosine_sim": cosine_sim
                }

                # Save plot only if SMAPE is above 15
                logger.info(f"Similarity Score: {similarity_score}")
                if smape < smape_threshold:
                    logger.info("Successfully found pair of tokens")
                    os.makedirs(token_folder_name, exist_ok=True)
                    plot_and_save(ts1=ts1_common,
                                  ts2=ts2_common,
                                  token1=token1,
                                  token2=token2,
                                  similarity_score=similarity_score,
                                  saving_file_path=saving_file_path)

                # Save the results in a CSV file
            with open(results_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                if token1 < token2:
                    token_pair = (token1, token2)
                else:
                    token_pair = (token2, token1)

                writer.writerow([token1, token2, dtw_distance, mae, rmse, mape, smape, token_pair])

            # Free memory if necessary
            del ts1_common, ts2_common, mae, rmse, mape, smape
            gc.collect()

        except Exception as e:
            print(f"Error processing tokens {token1} and {token2}: {e}")
            raise e

    def CoinCrossMappingSimilarity_multiprocessing(self,
                                                   results_with_cluster_id=None,
                                                   price_pivot_df=None,
                                                   files_path=None,
                                                   directory_names=None,
                                                   max_workers=2,
                                                   skip_noise_cluster=True,
                                                   smape_threshold=5):
        results_file = files_path['similarity_results_file_path']
        if not os.path.exists(results_file):
            with open(results_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    ['Token1', 'Token2', 'DTW_Distance', 'MAE', 'RMSE', 'MAPE', 'SMAPE', "TokenPair"])

        # Iterate through each cluster
        for cluster_id, cluster_group in tqdm(results_with_cluster_id.groupby('cluster')):
            if skip_noise_cluster == True:
                if cluster_id == -1:
                    continue

            logger.info(f"Processing Cluster ID: {cluster_id}")
            tokens = list(cluster_group['base_currency'].unique())
            logger.info(f'Tokens in this cluster : {tokens}')
            logger.info(f"Number of tokens in this cluster: {len(tokens)}")

            # Create a pool of processes
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, token1 in tqdm(enumerate(tokens), total=len(tokens), desc=f'Cluster {cluster_id}'):
                    for j, token2 in enumerate(tokens):
                        if i < j:  # Skip redundant calculations
                            result_df = pd.read_csv(files_path['similarity_results_file_path'])

                            token_pair = (token1, token2)
                            if result_df['TokenPair'].isin([token_pair]).shape[0] > 0:

                                df_row = result_df[result_df['TokenPair'] == token_pair]
                                if (df_row['SMAPE'] < smape_threshold).all():
                                    futures.append(
                                        executor.submit(self.process_pair,
                                                        token1,
                                                        token2,
                                                        price_pivot_df,
                                                        directory_names,
                                                        results_file,
                                                        smape_threshold))
                                else:
                                    logger.info(f"Skipping Token Pair : {token_pair}")
                                    continue

                # Wait for all processes to complete
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Token Pairs"):
                    future.result()
                    gc.collect()
            del cluster_group
            del tokens
            gc.collect()
        total_detected_similar_tokens = len(glob.glob(f"{directory_names['visualization_data_dir_name']}/*.png"))
        return total_detected_similar_tokens

    def ClustringBaseOnTokenNameTFIDF(self, df=None,
                                      max_features=1000,
                                      min_df=1,
                                      max_df=0.9,
                                      cluster_col_name='TokenNameBaseClusterLabel',
                                      save_plots=False,
                                      eps=0.5,
                                      min_cluster_size=2
                                      ):

        df.dropna(subset=['display_name'], inplace=True)  # Drop rows with NaN in 'display_name'
        df['display_name'] = df['display_name'].str.replace(r'[^a-zA-Z0-9]', '', regex=True).str.lower()

        vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df)

        X = vectorizer.fit_transform(df['display_name'])

        cosine_distance_matrix = pairwise_distances(X, metric='cosine')

        dbscan = DBSCAN(eps=eps, min_samples=min_cluster_size, metric='precomputed')  # Adjust eps based on your data
        labels = dbscan.fit_predict(cosine_distance_matrix)

        # Assign cluster labels back to the DataFrame
        df[f'{cluster_col_name}'] = labels
        if save_plots == True:
            try:
                generate_wordclouds_for_clusters(token_names_df=df,
                                                 directory_names=self.directory_names,
                                                 cluster_label_name=f'{cluster_col_name}')
            except Exception as e:
                logger.error(f"Error generating word clouds for cluster {cluster_col_name}: {e}")
                pass
        else:
            logger.info(f"Skipping cluster visualization: {cluster_col_name}")
        return df

    def __call__(self):
        testing_on_testing_ids = False
        TokenNameBaseClustring = False
        upsortoperation = False
        ClusterParameters = {
            "TokenBaseClustring": {
                "eps": 0.5,
                "min_samples": 2
            },
            "PriceLevelClusterParameters": {
                "eps": 5,
                "min_samples": 2
            },
            "SimilarityParameter": {
                "smape_threshold": 5,
            },
            "max_worker": 6,
            "use_previous_cluster_data_price_level": True,
            'visualization_of_price_cluster': False
        }

        logger.info('Reading Price data')
        raw_price_df = self.read_price_data(self.files_path['raw_price_data'])
        logger.info('Filtering Price Data')
        filtered_raw_price_df = self.filter_price_data(price_data=raw_price_df, price_points_threshold=100)

        logging.info(
            f"After Filtering Token id we have : {len(filtered_raw_price_df['base_currency'].unique().tolist())}")
        if testing_on_testing_ids:
            # testing_ids = list(filtered_raw_price_df['base_currency'].unique())[0:1000]
            testing_ids = [1166, 15390, 1146, 15467, 2012, 15593, 13049, 16668, 162796, 168956]
            random_ids = [75585, 53476, 18671, 60768, 21351, 96390, 61131, 46393, 21200, 20196, 21068,
                          85420, 21072, 160593, 61563, 2036, 1642, 27115, 46393, 18182, 23333, 18959,
                          21315, 19274, 19448, 20798, 21752, 20795, 20127, 21566, 21583]
            testing_ids = testing_ids + random_ids
            logger.info(f"Number of Testing ids: {len(testing_ids)}")
            filtered_raw_price_df = filtered_raw_price_df[filtered_raw_price_df['base_currency'].isin(testing_ids)]

        filtered_raw_price_df[filtered_raw_price_df['base_currency'] == 15467].to_csv('testing_sample.csv', index=False)
        total_number_of_tokens = len(filtered_raw_price_df['base_currency'].unique())
        logger.info(f"total number of tokens: {total_number_of_tokens}")

        logger.info('Reading Token Data')
        token_names_df = pd.read_csv(self.files_path['raw_token_names'])

        logger.info('Filtering Token Data')
        token_names_df = self.filter_token_names(price_data=filtered_raw_price_df, token_data=token_names_df)


        if upsortoperation==True:
            pinecone_object = PineconeStorage(key=config['PINECONE_API_KEY'],
                                              save_embeddings_root_path=self.directory_names['embedding_root_path']
)
            index_name = "cross-mapping"
            # PineconeStorage_storage(index_name=index_name,data=filtered_raw_price_df)
            pinecone_object.create_index(index_name=index_name)
            index_object = pinecone_object.get_index(index_name)

            # self.is_record_in_vec_db(index=index_object,record_id=803)

            pinecone_object.insert_data(index=index_object,
                       data = token_names_df
                                        )
            return








        token_names_df.to_csv(f"{self.directory_names['preprocessed_data_dir_name']}/token_names_filtered.csv", index=False)
        if TokenNameBaseClustring == True:
            logger.info(f"Current Parameters : {ClusterParameters}")
            logger.info('Token Name Base Clustering Started ...')
            token_names_df = self.ClustringBaseOnTokenName(df=token_names_df,
                                                           cluster_col_name='TokenNameBaseClusterLabel',
                                                           save_plots=False,
                                                           eps=ClusterParameters['TokenBaseClustring']['eps'],
                                                           min_cluster_size=ClusterParameters['TokenBaseClustring'][
                                                               'min_samples'],

                                                           )

            total_cluster_base_on_token_name = len(list(token_names_df['TokenNameBaseClusterLabel'].unique()))
            logger.info(f"Total Number of cluster base on token name : {total_cluster_base_on_token_name}")
            for cluster_id, cluster_group in token_names_df.groupby('TokenNameBaseClusterLabel'):
                if cluster_id == -1:
                    continue
                logger.info(f"Processing Cluster ID of TokenNameBase: {cluster_id}")
                # print(list(cluster_group))
                tokens = list(cluster_group['id'].unique())
                # logger.info(f'Tokens in this cluster : {tokens}')
                logger.info(f"Number of tokens in this cluster: {len(tokens)}")

                logger.info('Merging Token Name Data to price data after Clustering')
                merged_df = filtered_raw_price_df.merge(cluster_group, left_on=['base_currency'], right_on=['id'])

                logger.info(f'Pivoting the data')
                price_pivot = merged_df.pivot_table(index='timestamp_utc', columns='base_currency', values='open')
                price_pivot = price_pivot.fillna(method='ffill').fillna(method='bfill')

                # Standardize the pivoted data
                logger.info(f'Applying Scaling on price data')
                scaler = StandardScaler()
                price_scaled = scaler.fit_transform(price_pivot.T)

                # Step 4: DBSCAN clustering on time series data
                logger.info("Performing DBSCAN on time series data...")
                dbscan = DBSCAN(eps=ClusterParameters['PriceLevelClusterParameters']['eps'],
                                min_samples=ClusterParameters['PriceLevelClusterParameters']['min_samples'],
                                metric=smape_distance_metric,
                                n_jobs=5)
                labels = dbscan.fit_predict(price_pivot.T.values)

                # Store clustering results
                cluster_results = pd.DataFrame({'base_currency': price_pivot.columns, 'cluster': labels})
                # cluster_results.to_csv('cluster_results_time_series.csv',index=False)

                # cluster_results = pd.read_csv('cluster_results_time_series.csv')
                logger.info(f"Created DataFrame for cluster results. Shape: {cluster_results.shape}")

                # Merge results with the original filtered DataFrame
                results_with_cluster_id = pd.merge(merged_df, cluster_results, on='base_currency')
                results_with_cluster_id['timestamp_utc'] = pd.to_datetime(results_with_cluster_id['timestamp_utc'])
                results_with_cluster_id = results_with_cluster_id.sort_values(by='timestamp_utc')

                # Visualization (if needed)
                # cluster_visualization_of_time_series(results_with_cluster_id=results_with_cluster_id,
                #                                      cluster_dir_path=self.directory_names['cluster_dir_path'])

                # Create pivot table from results with cluster IDs
                price_pivot_df = results_with_cluster_id.pivot_table(index='timestamp_utc',
                                                                     columns='base_currency',
                                                                     values='open')

                # Calculate similar tokens using multiprocessing
                total_similar_tokens = self.CoinCrossMappingSimilarity_multiprocessing(
                    results_with_cluster_id, price_pivot_df, self.files_path, self.directory_names,
                    max_workers=ClusterParameters['max_worker'],
                    skip_noise_cluster=True,
                    smape_threshold=ClusterParameters[
                        'SimilarityParameter'][
                        'smape_threshold']
                )
                logger.info(f"Total similar tokens: {total_similar_tokens}")
                gc.collect()

        else:
            logger.info(f"Current Parameters : {ClusterParameters}")

            logger.info('Merging price data and token data')
            merged_df = filtered_raw_price_df.merge(token_names_df, left_on=['base_currency'], right_on=['id'])

            # sample_for_testing
            logger.info('Separating Testing currency id')
            # testing_base_currency_id = list(merged_df['base_currency'].unique())[0:50]
            # testing_base_currency_id = [1166,15390,1146,15467,2012,15593,13049,16668,162796,168956]

            # merged_df =  merged_df [merged_df['base_currency'].isin(testing_base_currency_id)]

            logger.info(f'Pivoting the data')
            price_pivot = merged_df.pivot_table(index='timestamp_utc', columns='base_currency', values='open')
            price_pivot = price_pivot.fillna(method='ffill').fillna(method='bfill')

            if ClusterParameters['use_previous_cluster_data_price_level']:
                cluster_price_level_file_path = self.files_path['price_level_clustered_result_file_path']
                logger.info("Loading the Price Cluster data file....")
                cluster_results = pd.read_csv(cluster_price_level_file_path)

            else:
                # Standardize the pivoted data
                logger.info(f'Appling Scaling on price data')
                # scaler = StandardScaler()
                # price_scaled = scaler.fit_transform(price_pivot.T)

                # Step 4: DBSCAN clustering on time series data
                logger.info("Performing DBSCAN on time series data...")
                dbscan = DBSCAN(eps=ClusterParameters['PriceLevelClusterParameters']['eps'],
                                min_samples=ClusterParameters['PriceLevelClusterParameters']['min_samples'],
                                metric=smape_distance_metric,
                                n_jobs=2)

                labels = dbscan.fit_predict(price_pivot.T.values)
                cluster_results = pd.DataFrame({'base_currency': price_pivot.columns, 'cluster': labels})

                # Store clustering results
                cluster_results.to_csv(self.files_path['price_level_clustered_result_file_path'], index=False)

            del raw_price_df
            del filtered_raw_price_df
            del token_names_df
            del price_pivot
            gc.collect()

            logger.info(f"Created DataFrame for cluster results. Shape: {cluster_results.shape}")

            # Merge results with the original filtered DataFrame
            results_with_cluster_id = pd.merge(merged_df, cluster_results, on='base_currency')

            del cluster_results
            gc.collect()

            results_with_cluster_id['timestamp_utc'] = pd.to_datetime(results_with_cluster_id['timestamp_utc'])
            results_with_cluster_id = results_with_cluster_id.sort_values(by='timestamp_utc')

            if ClusterParameters['visualization_of_price_cluster']:
                # Visualization (if needed)
                cluster_visualization_of_time_series(results_with_cluster_id=results_with_cluster_id,
                                                     cluster_dir_path=self.directory_names['cluster_dir_path'])

            # Create pivot table from results with cluster IDs
            price_pivot_df = results_with_cluster_id.pivot_table(index='timestamp_utc', columns='base_currency',
                                                                 values='open')

            # Calculate similar tokens using multiprocessing
            total_similar_tokens = self.CoinCrossMappingSimilarity_multiprocessing(results_with_cluster_id,
                                                                                   price_pivot_df,
                                                                                   self.files_path,
                                                                                   self.directory_names,
                                                                                   max_workers=ClusterParameters[
                                                                                       'max_worker'],
                                                                                   skip_noise_cluster=True,
                                                                                   smape_threshold=ClusterParameters[
                                                                                       'SimilarityParameter'][
                                                                                       'smape_threshold'])
            logger.info(f"Total similar tokens: {total_similar_tokens}")
            gc.collect()


if __name__ == '__main__':
    CrossMappingObject = CrossMapping()
    CrossMappingObject()
