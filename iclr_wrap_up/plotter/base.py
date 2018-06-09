import matplotlib


class BasePlotter:
    """Base class for plotters."""
    plotname = ''

    def generate(self, measures_summary):
        fig = self.plot(measures_summary)
        filename = f'plots/{self.plotname}.png'
        fig.savefig(filename, bbox_inches='tight', dpi=600)

        self.run.add_artifact(filename, name=self.plotname)

    def plot(self, measures_summary) -> matplotlib.figure.Figure:
        raise NotImplementedError
