import logging
import pinecone
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from App.adoptors.embeddings_client import get_embeddings
import json


class PineconeStorage:
    def __init__(self, key=None, save_embeddings_root_path=None):
        self.pinecone_api_key = key
        self.is_connected, self.pinecone = self.connect_pinecone()
        self.save_embeddings_root_path = save_embeddings_root_path

    def create_index(self, index_name):
        try:

            self.pinecone.create_index(
                name=index_name,
                dimension=1536,
                metric='euclidean',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1')
            )
            return True
        except Exception as e:
            return False

    def delete_index(self, index_name):
        active_indexes = pinecone.list_indexes()
        if index_name in active_indexes:
            logging.info(f"DELETING index {index_name}")
            pinecone.delete_index(index_name)

    def index_exists(self, index_name):
        try:
            active_indexes = pinecone.list_indexes()
        except Exception as e:
            logging.error(e)
            return False
        return index_name in active_indexes

    def get_index(self, index_name):
        index = self.pinecone.Index(index_name)
        return index

    def connect_pinecone(self):
        try:
            pc = Pinecone(api_key=self.pinecone_api_key)
            return True, pc
        except Exception as e:
            logging.info("Error in connected to VectorDB ")
            return False, pc

    def is_record_in_vec_db(self, index=None, record_id=None):
        if self.is_connected:
            record = index.query(namespace="cross", id=f"{record_id}", top_k=1, include_values=False)
            matches = record['matches']
            print("matches : ", matches)
            print("len of matches : ", len(matches))
            if len(matches) == 0:
                return False
            else:
                return True
            print(f"records : {record}")

    def insert_data(self, index=None, data=None, inserted_ids=None, namespace_name=None, batch_size=8):

        for row_index, row in tqdm(data.iterrows(), total=len(data), desc="Inserting rows"):
            logging.info(f"Inserting row: {row['display_name']}")
            is_in_vector_db = self.is_record_in_vec_db(index=index, record_id=row["id"])
            if is_in_vector_db == True:
                print("Skipping embeddings")
                continue
            embedding_vector = get_embeddings(texts=row['display_name'])

            # Save embedding to JSON file
            embedding_json = {
                "id": str(row["id"]),
                "embedding": embedding_vector[0]['embedding'],
                "metadata": {
                    "full_name": row['full_name'],
                    "display_name": row['display_name'],
                    "TokenID": row["id"]
                }
            }

            # Save the embedding to a JSON file (one per row)
            with open(f"{self.save_embeddings_root_path}/{row['id']}_embedding.json", 'w') as f:
                json.dump(embedding_json, f)

            # Upsert the vector into Pinecone
            vector = {
                "id": str(row["id"]),
                "values": embedding_vector[0]['embedding'],
                "metadata": embedding_json["metadata"]
            }

            index.upsert(vectors=[vector], namespace="cross")
            logging.info(f"Upserted vector with id '{row['display_name']}'")

    def delete_doc(self, index, doc_id, namespace_name):
        try:
            index.delete(
                filter={
                    "doc_id": {"$eq": doc_id},
                },
                namespace=namespace_name,
            )
            return 1
        except Exception as e:
            logging.error(e)
            return 0

    def get_matches(self, index_name=None, token_id=None, namespace=None, embedding_vector=None, query_filter=None,
                    size=None):
        if self.is_connected:
            index = self.pinecone.Index(index_name)

            try:
                # Check if embedding is available in the local JSON file
                embedding_json_file_path = f"{self.save_embeddings_root_path}/{token_id}_embedding.json"

                try:
                    with open(embedding_json_file_path, 'r') as file:
                        logging.info(f"Loading embedding from local JSON file: {embedding_json_file_path}")
                        embedding_data = json.load(file)
                        embedding_vector = embedding_data['embedding']
                except FileNotFoundError:
                    logging.error(
                        f"Embedding JSON file not found at {embedding_json_file_path}. Proceeding with Pinecone query.")

                    # If file not found, query Pinecone as usual
                    response = index.query(
                        vector=embedding_vector,
                        top_k=size,
                        include_values=False,
                        include_metadata=True,
                        namespace=namespace,
                        filter=query_filter
                    )
                    return {'matches': response['matches']}

                # Query Pinecone using the loaded embedding from the JSON file
                response = index.query(
                    vector=embedding_vector,
                    top_k=size,
                    include_values=False,
                    include_metadata=True,
                    namespace=namespace,
                    filter=query_filter
                )
                return {'matches': response['matches']}

            except Exception as e:
                logging.error(f"Error querying Pinecone: {e}")
                return {"matches": []}
        else:
            logging.error("Not connected to Pinecone.")
            return {"matches": []}

    def __call__(self, index_name="cross-mapping", data=None):

        self.create_index(index_name=index_name)
        index_object = self.get_index(index_name)

        # self.is_record_in_vec_db(index=index_object,record_id=803)

        self.insert_data(index=index_object,
                         data=data)

        # matches = self.get_matches(
        #     index_name="cross-mapping",
        #     token_id = 23577  ,
        #     namespace="cross",
        #                 # embedding_vector =  get_embeddings(texts='Rupees')[0]['embedding'],
        #                  query_filter=None,
        #                 size = 5
        #                 )

        # print(f"matches : {matches}")

        pass


if __name__ == "__main__":
    pass
    # PineconeStorage_storage = PineconeStorage()
    # index_name = "cross-mapping"
    #
    # PineconeStorage_storage(index_name=index_name,
    #                        data=df)
