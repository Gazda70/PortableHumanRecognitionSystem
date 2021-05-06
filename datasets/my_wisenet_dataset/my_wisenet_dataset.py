"""my_wisenet_dataset dataset."""

import tensorflow_datasets as tfds
import glob


# TODO(my_wisenet_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(my_wisenet_dataset): BibTeX citation
_CITATION = """
"""

PATH_TO_DATA = 'E:/PortableHumanRecognitionSystem/PortableHumanRecognitionSystem/datasets/my_wisenet_dataset/test_set.zip'

class MyWisenetDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my_wisenet_dataset dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(my_wisenet_dataset): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(None, None, 3)),
                'label': tfds.features.ClassLabel(names=['no', 'yes']),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # Set to `None` to disable
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(my_wisenet_dataset): Downloads the data and defines the splits
        path = dl_manager.extract(PATH_TO_DATA)

        # TODO(my_wisenet_dataset): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(path / 'train_set'),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(my_wisenet_dataset): Yields (key, example) tuples from the dataset

        # TODO tutaj powinno być path.glob(*.typ_pliku), ale na razie nie dzialalo wiec jest hardcode - naprawic
        for f in glob.glob('E:\PortableHumanRecognitionSystem\PortableHumanRecognitionSystem\datasets\my_wisenet_dataset\\test_set\\train_set\*.jpeg'):
            yield 'key', {
                'image': f,
                'label': 'yes',
            }

