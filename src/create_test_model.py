import os

import numpy as np
from ms2deepscore import SettingsMS2Deepscore
from ms2deepscore.train_new_model.train_ms2deepscore import train_ms2ds_model
from tests.create_test_spectra import pesticides_test_spectra
from tests.test_data_generators import create_test_spectra

from ms2deepscore_java_conversion.src.save_to_torch_script import save_model_for_java_predict_embeddings


def create_test_model_for_java(results_folder):
    """Creates a test model. This model is used for running CI tests in MZMine."""
    spectra = create_test_spectra(8)
    settings = SettingsMS2Deepscore(**{
        "mz_bin_width": 1.0,
        "epochs": 2,  # to speed up tests --> usually many more
        "base_dims": (100, 100),  # to speed up tests --> usually larger
        "embedding_dim": 50,  # to speed up tests --> usually larger
        "same_prob_bins": np.array([(0, 0.5), (0.5, 1.0)]),
        "average_pairs_per_bin": 2,
        "batch_size": 8,
        "additional_metadata": [("StandardScaler", {"metadata_field": "precursor_mz",
                                                    "mean": 0.0,
                                                    "standard_deviation": 1000.0}),
                                ("CategoricalToBinary", {"metadata_field": "ionmode",
                                                         "entries_becoming_one": "positive",
                                                         "entries_becoming_zero": "negative"})],
    })

    model_file_name = os.path.join(results_folder, "ms2deepscore_model.pt")
    train_ms2ds_model(spectra, pesticides_test_spectra(), results_folder, settings)

    save_model_for_java_predict_embeddings(model_file_name, os.path.join(results_folder, "java_ms2deepscore_model.pt"))