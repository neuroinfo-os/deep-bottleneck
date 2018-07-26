from bson import ObjectId
import pandas as pd
from typing import *

from iclr_wrap_up.eval_tools import artifact


class Experiment:
    artifact_name_to_cls = {
        'infoplane': artifact.PNGArtifact,
        'snr': artifact.PNGArtifact,
        'infoplane_movie': artifact.MP4Artifact,
        'information_measures': artifact.CSVArtifact,
        'activations': artifact.PNGArtifact,
        'single_neuron_activations': artifact.PNGArtifact
    }

    def __init__(self, id_, database, grid_filesystem, config, artifact_links, metric_links):
        self.id = id_
        self.config = config
        self._artifacts_links = artifact_links
        self._metrics_links = metric_links
        self._database = database
        self._grid_filesystem = grid_filesystem
        self._artifacts = None
        self._metrics = None

    def __repr__(self):
        return f'{self.__class__.__name__}(id={self.id})'

    @classmethod
    def from_db_object(cls, database, grid_filesystem, experiment_data: dict):
        config = experiment_data['config']
        artifacts_links = experiment_data['artifacts']
        metric_links = experiment_data['info']['metrics']
        id_ = experiment_data['_id']
        return cls(id_, database, grid_filesystem, config, artifacts_links, metric_links)

    @property
    def artifacts(self) -> Dict[str, artifact.Artifact]:
        """
        The artifacts belonging to the experiment.

        Returns:
            A mapping from artifact names to artifact objects, that
            belong to the experiment.
        """
        if self._artifacts is None:
            self._artifacts = self._load_artifacts()

        return self._artifacts

    @property
    def metrics(self) -> Dict[str, pd.Series]:
        """
        The metrics belonging to the experiment.

        Returns:
            A mapping from metric names to pandas Series objects, that
            belong to the experiment.
        """
        if self._metrics is None:
            self._metrics = self._load_metrics()

        return self._metrics

    def _load_artifacts(self):
        artifacts = {
            artifact['name']: self.artifact_name_to_cls[artifact['name']](
                artifact['name'], self._grid_filesystem.get(artifact['file_id'])
            )
            for artifact in self._artifacts_links}
        return artifacts

    def _load_metrics(self):
        metrics = {}
        for metric_link in self._metrics_links:
            metric_db_entry = self._database.metrics.find_one({'_id': ObjectId(metric_link['id'])})
            metrics[metric_link['name']] = pd.Series(data=metric_db_entry['values'],
                                                     index=pd.Index(metric_db_entry['steps'], name='step'),
                                                     name=metric_link['name'])
        return metrics
