MONGODB_ADMINUSERNAME = '<MONGO_INITDB_ROOT_USERNAME>'
MONGODB_ADMINPASSWORD = '<MONGO_INITDB_ROOT_PASSWORD>'
MONGODB_HOST = '<server_ip_address>:27017'
MONGODB_DBNAME = '<MONGO_DATABASE>'
MONGODB_URI = f'mongodb://{MONGODB_ADMINUSERNAME}:{MONGODB_ADMINPASSWORD}@{MONGODB_HOST}/?authMechanism=SCRAM-SHA-1'