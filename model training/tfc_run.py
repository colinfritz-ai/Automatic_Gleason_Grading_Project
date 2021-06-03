import tensorflow_cloud as tfc
tfc.run(entry_point='/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/train_production_model.py',
	    requirements_txt='/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/tfc_scripts/requirements.txt',
	    docker_config="auto",
	    distribution_strategy="auto",
	    chief_config=tfc.MachineConfig(
        cpu_cores=8,
        memory=30,
        accelerator_type=tfc.AcceleratorType.NVIDIA_TESLA_T4,
        accelerator_count=2,
    	),
	    worker_config="auto",
	    worker_count=0,
	    entry_point_args=None,
	    stream_logs=False,
	    job_labels=None,
	    service_account= "automated-gleason-grading--644@tf2servetutorial.iam.gserviceaccount.com",
		)
# def implicit():
#     from google.cloud import storage

#     # If you don't specify credentials when constructing the client, the
#     # client library will look for credentials in the environment.
#     storage_client = storage.Client()

#     # Make an authenticated API request
#     buckets = list(storage_client.list_buckets())
#     print(buckets)

# implicit()




