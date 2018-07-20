from pymongo import MongoClient
import gridfs

from iclr_wrap_up import credentials
from bson import ObjectId
import pandas as pd
from functools import lru_cache

from iclr_wrap_up.eval_tools.experiment import Experiment

from typing import *


class ExperimentLoader:
    """Loads artifacts related to experiments."""

    def __init__(self, mongo_uri=credentials.MONGODB_URI, db_name=credentials.MONGODB_DBNAME):
        client = MongoClient(mongo_uri)
        self.db = client[db_name]
        self.runs = self.db.runs
        self.fs = gridfs.GridFS(self.db)

    def find_by_ids(self, experiment_ids: Iterable[int]) -> List[Experiment]:
        """
        Find experiments based on a collection of ids.

        Args:
            experiment_ids: Iterable of experiment ids.

        Returns:
            The experiments corresponding to the ids.
        """
        experiments = [self.find_by_id(experiment_id) for experiment_id in experiment_ids]

        return experiments

    # The cache makes sure that both retrieval of the experiments
    # is not unnecessarily done more than once.
    @lru_cache(maxsize=32)
    def find_by_id(self, experiment_id: int) -> Experiment:
        """
        Find experiment based on its id.

        Args:
            experiment_id: The id  of the experiment.

        Returns:
            The experiment corresponing to the id.
        """
        experiment = self._find_experiment(experiment_id)

        return self._make_experiment(experiment)

    @lru_cache(maxsize=32)
    def find_by_name(self, name: str) -> List[Experiment]:
        """
        Find experiments based on regex search against its name.

        A partial match between experiment name and regex is enough
        to find the experiment.

        Args:
            name: Regex that is matched against the experiment name.

        Returns:
            The matched experiments.
        """
        return self.find_by_config_key('experiment.name', name)

    @lru_cache(maxsize=32)
    def find_by_config_key(self, key: str, value: str):
        """
        Find experiments based on regex search against an configuration value.

        A partial match between configuration value and regex is enough
        to find the experiment.

        Args:
            name: Regex that is matched against the experiment's configuration.

        Returns:
            The matched experiments.
        """
        cursor = self.runs.find({key: {'$regex': rf'{value}'}})
        experiments = [self._make_experiment(experiment) for experiment in cursor]
        return experiments

    @lru_cache(maxsize=32)
    def _find_experiment(self, experiment_id: int):
        return self.runs.find_one({'_id': experiment_id})

    def _make_experiment(self, experiment):
        return Experiment.from_db_object(self.db, self.fs, experiment)
