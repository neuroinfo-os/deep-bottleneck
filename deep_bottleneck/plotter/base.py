import matplotlib


class BasePlotter:
    """Base class for plotters."""
    plotname = ''
    file_ext = ''

    def make_filename(self, suffix):
        suffix = '_' + suffix if suffix else suffix
        filename = f'plots/{self.plotname}{suffix}.{self.file_ext}'
        return filename

    def generate(self, measures_summary, suffix=''):
        fig = self.plot(measures_summary)
        filename = self.make_filename(suffix)
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        suffix = '_' + suffix if suffix else suffix
        artifact_name = f'{self.plotname}{suffix}'

        self.run.add_artifact(filename)

    def plot(self, measures_summary) -> matplotlib.figure.Figure:
        raise NotImplementedError
