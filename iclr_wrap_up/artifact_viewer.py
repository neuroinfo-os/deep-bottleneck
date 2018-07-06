from io import BytesIO
from pymongo import MongoClient
import gridfs
import matplotlib.pyplot as plt
import numpy as np
from iclr_wrap_up import credentials
from IPython.display import HTML
import pandas as pd


class ArtifactLoader:
    """Loads artifacts related to experiments."""

    def __init__(self, mongo_uri=credentials.MONGODB_URI, db_name=credentials.MONGODB_DBNAME):
        client = MongoClient(mongo_uri)
        db = client[db_name]
        self.runs = db.runs
        self.fs = gridfs.GridFS(db)
        self.mapping = {'infoplane': PNGArtifact, 'snr': PNGArtifact, 'infoplane_movie': MP4Artifact,
                        'information_measures': CSVArtifact, 'activations': PNGArtifact}
        
    def load(self, experiment_id: int):
        experiment = self.runs.find_one({'_id': experiment_id})
        val_acc = experiment["captured_out"][experiment["captured_out"].rfind("val_acc")+9:]
        val_acc = val_acc[:val_acc.find(" ")]; val_acc = val_acc[:val_acc.find("\n")]; val_acc = float(val_acc)
        artifacts = {artifact['name']: self.mapping[artifact['name']](artifact['name'], self.fs.get(artifact['file_id']))
                     for artifact in experiment['artifacts']}
        artifacts['name'] = experiment["experiment"]["name"]
        artifacts['val_acc'] = val_acc
        return artifacts
        

class Artifact:
    """Displays or saves an artifact."""

    extension = ""

    def __init__(self, name, file):
        self.name = name
        self.file = file
        self.content = None

    def __repr__(self):
        return f'{self.__class__}(name={self.name})'

    def save(self):
        self._read()
        with open(self._make_filename(), 'wb') as file:
            file.write(self.content)

    def _read(self):
        if self.content is None:
            self.content = self.file.read()

    def _make_filename(self):
        parts = self.file.filename.split('/')
        return f'{parts[-2]}_{parts[-1]}.{self.extension}'


class PNGArtifact(Artifact):
    """Displays or saves a PNG artifact."""

    extension = "png"

    def __init__(self, name, file):
        super().__init__(name, file)
        self.fig = None

    def show(self, figsize=(10, 10)):
        if self.fig is None:
            self._make_figure(figsize)
        return self.fig

    def _make_figure(self, figsize):
        self._read()
        self.fig, ax = plt.subplots(figsize=figsize)
        img = plt.imread(BytesIO(self.content))
        ax.imshow(img)
        ax.axis('off')


class MP4Artifact(Artifact):
    """Displays or saves a MP4 artifact"""

    extension = "mp4"

    def __init__(self, name, file):
        super().__init__(name, file)
        self.movie = None

    def show(self):
        if self.movie is None:
            self._make_movie()
        return self.movie

    def _make_movie(self):
        self.save()
        self.movie = HTML(f"""
        <video width="640" height="480" controls autoplay>
          <source src="{self._make_filename()}" type="video/mp4">
        </video>
        """)

class CSVArtifact(Artifact):
    """Displays and saves a CSV artifact"""

    extension = "csv"

    def __init__(self, name, file):
        super().__init__(name, file)
        self.df = None

    def _make_df(self):
        self.df = pd.read_csv(self.file)

    def show(self):
        if self.df is None:
            self._make_df()
        return self.df

