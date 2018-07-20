from io import BytesIO
import matplotlib.pyplot as plt
from IPython.display import HTML
import pandas as pd


class Artifact:
    """Displays or saves an artifact."""

    extension = ""

    def __init__(self, name, file):
        self.name = name
        self.file = file
        self._content = None

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name})'

    def save(self):
        with open(self._make_filename(), 'wb') as file:
            file.write(self.content)

    @property
    def content(self):
        if self._content is None:
            self._content = self.file.read()
        return self._content

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

    def show(self):
        if self.df is None:
            self.df = self._make_df()
        return self.df

    def _make_df(self):
        df = pd.read_csv(self.file)
        return df
