from pymongo import MongoClient
import gridfs

from iclr_wrap_up import credentials
from bson import ObjectId
import pandas as pd
from functools import lru_cache

from iclr_wrap_up import artifact_viewer

from typing import *


class ExperimentLoader:
    """Loads artifacts related to experiments."""

    def __init__(self, mongo_uri=credentials.MONGODB_URI, db_name=credentials.MONGODB_DBNAME):
        client = MongoClient(mongo_uri)
        self.db = client[db_name]
        self.runs = self.db.runs
        self.fs = gridfs.GridFS(self.db)

    # The cache makes sure that both retrieval of the artifacts and
    # their content is not unnecessarily done more than once.
    @lru_cache(maxsize=32)
    def find_by_id(self, experiment_id: int):
        experiment = self._find_experiment(experiment_id)

        return self._make_experiment(experiment)

    @lru_cache(maxsize=32)
    def find_by_name(self, name):
        return self.find_by_config_key('experiment.name', name)

    @lru_cache(maxsize=32)
    def find_by_config_key(self, key, value):
        cursor = self.runs.find({key: {rf'{value}'}})
        experiments = [self._make_experiment(experiment) for experiment in cursor]
        return experiments

    @lru_cache(maxsize=32)
    def _find_experiment(self, experiment_id: int):
        return self.runs.find_one({'_id': experiment_id})

    def _make_experiment(self, experiment):
        return Experiment.from_db_object(self.db, self.fs, experiment)


class Experiment:
    artifact_name_to_cls = {
        'infoplane': artifact_viewer.PNGArtifact,
        'snr': artifact_viewer.PNGArtifact,
        'infoplane_movie': artifact_viewer.MP4Artifact,
        'information_measures': artifact_viewer.CSVArtifact,
        'activations': artifact_viewer.PNGArtifact}

    def __init__(self, id_, db, fs, config, artifact_links, metric_links):
        self.id = id_
        self.config = config
        self._artifacts_links = artifact_links
        self._metrics_links = metric_links
        self._db = db
        self._fs = fs
        self._artifacts = None
        self._metrics = None

    def __repr__(self):
        return f'{self.__class__.__name__}(id={self.id})'

    @classmethod
    def from_db_object(cls, db, fs, experiment_data: dict):
        config = experiment_data['config']
        artifacts_links = experiment_data['artifacts']
        metric_links = experiment_data['info']['metrics']
        id_ = experiment_data['_id']
        return cls(id_, db, fs, config, artifacts_links, metric_links)

    @property
    def artifacts(self) -> Dict[str, artifact_viewer.Artifact]:
        if self._artifacts is None:
            self._artifacts = self._load_artifacts()

        return self._artifacts

    @property
    def metrics(self) -> Dict[str, pd.Series]:
        if self._metrics is None:
            self._metrics = self._load_metrics()

        return self._metrics

    def _load_artifacts(self):
        artifacts = {
            artifact['name']: self.artifact_name_to_cls[artifact['name']](
                artifact['name'], self._fs.get(artifact['file_id'])
            )
            for artifact in self._artifacts_links}
        return artifacts

    def _load_metrics(self):
        metrics = {}
        for metric_link in self._metrics_links:
            metric_db_entry = self._db.metrics.find_one({'_id': ObjectId(metric_link['id'])})
            metrics[metric_link['name']] = pd.Series(data=metric_db_entry['values'],
                                                     index=pd.Index(metric_db_entry['steps'], name='step'),
                                                     name=metric_link['name'])
        return metrics
