from google.cloud import storage
import os

@hydra.main(config_path="config.yaml")
def retrieve(cfg):
    # NOTE: GOOGLE_APPLICATION_CREDENTIALS env var must be set
    storage_client = storage.Client()
    buckets = cfg.data.google_storage.keys()
    force = cfg.data.google_storage.force
    for bucket_name in buckets:
        for blob_name, destination in buckets[bucket_name].items():
            if force or not os.path.exists(destination):
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                blob.download_to_filename(destination)

if __name__ == "__main__":
    retrieve()