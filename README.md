[![Documentation Status](https://readthedocs.org/projects/deep-bottleneck/badge/?version=latest)](http://deep-bottleneck.readthedocs.io/en/latest/?badge=latest)
[![Build status](https://travis-ci.com/neuroinfo-os/deep-bottleneck.svg?branch=master)](https://travis-ci.com/neuroinfo-os/deep-bottleneck)

## Documentation

To build the documentation locally run

    $ cd docs
    $ make html

The generated files will be found in the `build/` directory. 


## MongoDB
For using sarcred MongoDB instance is required. To start a new instance on 
a server with docker run

    $ docker run --name my_mongo -d -p 27017:27017 mongo
    
or better with express web frontend and authentification

    $ infrastructure
    # Adapt username and password
    $ docker-compose up

Then put the IP address of your server and the credentials into the MongoObserver of `experiment.py`.


To connect with sacredboard run

    $ sacredboard -m <IPADDRESS>:27017:<DBNAME>
    
or with authentification

    $ sacredboard -mu mongodb://<user>:<pwd>@<host>/?authMechanism=SCRAM-SHA-1 <db_name> 
