from sacred import Experiment
from sacred.observers import MongoObserver
from . import credentials
from pymongo import MongoClient
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds


def test_sacred():
    SETTINGS.CAPTURE_MODE = 'sys'

    ex = Experiment('test_experiment')
    ex.captured_out_filter = apply_backspaces_and_linefeeds
    ex.observers.append(MongoObserver.create(url=credentials.MONGODB_URI,
                                             db_name=credentials.MONGODB_DBNAME))
    text = 'test captured output'

    @ex.main
    def conduct():
        print(text)

    run = ex.run()

    client = MongoClient(credentials.MONGODB_URI)
    db = client[credentials.MONGODB_DBNAME]
    db_run = db.runs.find_one({'_id': run._id})

    assert db_run['captured_out'] == text + '\n'
