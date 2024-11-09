
class VectorEmbedddings:


	def __init__():
		pass

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

	def ClustringBaseOnTokenNameLLM(self, df=None,
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

    def save_embedding(self, token_names_df=None, max_workers=2, batch_size=50):


        total_tokens = token_names_df.shape[0]

        # Use ThreadPoolExecutor for multithreading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for start in tqdm(range(0, total_tokens, batch_size), desc="Generating Embeddings"):
                end = min(start + batch_size, total_tokens)
                display_names = token_names_df['display_name'].iloc[start:end].tolist()
                display_names = token_names_df['display_name'].iloc[start:end].tolist()
                
                # Submit the task to the executor
                futures.append(executor.submit(self.process_embedding_batch, display_names,))

                # Rate limiting
                if len(futures) >= 2100:  # Limit based on API calls
                    for future in tqdm(futures, desc="Waiting for results"):
                        embeddings = future.result()
                        for i, embedding in enumerate(embeddings):
                            if embedding is not None:
                                token_names_df.at[start + i, 'embedding'] = embedding
                    futures.clear()  # Clear the completed futures

                    # Sleep to respect rate limits
                    time.sleep(30)  # Adjust as necessary based on your rate limits

            # Collect any remaining results
            for future in tqdm(futures, desc="Collecting Remaining Results"):
                embeddings = future.result()
                for i, embedding in enumerate(embeddings):
                    if embedding is not None:
                        token_names_df.at[start + i, 'embedding'] = embedding

        token_names_df.to_csv(save_path, index=False)
        logger.info("Embedding generation completed and saved.")

    
	

	def __call__():
		pass