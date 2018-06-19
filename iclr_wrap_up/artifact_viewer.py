from io import BytesIO
from pymongo import MongoClient
import gridfs
import matplotlib.pyplot as plt
import numpy as np
from iclr_wrap_up import credentials


class ArtifactLoader:
    """Loads artifacts related to experiments."""
    def __init__(self, mongo_uri=credentials.MONGODB_URI, db_name=credentials.MONGODB_DBNAME):
        client = MongoClient(mongo_uri)
        db = client[db_name]
        self.runs = db.runs
        self.fs = gridfs.GridFS(db)
        
    def load(self, experiment_id: int):
        experiment = self.runs.find_one({'_id': experiment_id})
        artifacts = {artifact['name']: Artifact(artifact['name'], self.fs.get(artifact['file_id']))
                     for artifact in experiment['artifacts']}
        return artifacts
        

class Artifact:
    """Displays or saves an artifact."""
    def __init__(self, name, file):
        self.name = name
        self.file = file
        self.content = None
        self.fig = None
    
    def __repr__(self):
        return f'Artifact(name={self.name})'

    def show(self):
        try:
            if self.fig is None:
                self._make_figure()
            return self.fig
        except:
            raise ValueError('Something went wrong. Is the artifact a png file?')

    def save(self):
        self._read()
        with open(self._make_filename(), 'wb') as file:
            file.write(self.content)

    def _read(self):
        if self.content is None:
            self.content = self.file.read()

    def _make_figure(self):
        self._read()
        self.fig, ax = plt.subplots(figsize=(10, 10))
        img = plt.imread(BytesIO(self.content))
        ax.imshow(img)
        ax.axis('off')

    def _make_filename(self):
        parts = self.file.filename.split('/')
        return f'{parts[-2]}_{parts[-1]}.png'
