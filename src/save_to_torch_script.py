import torch

from ms2deepscore.models import load_model


def save_model_for_java_predict_embeddings(input_model_file_name, output_model_file_name):
    """Saves a model in a file format compatible with java.
    The model will predict embeddings from binned spectra"""
    model = load_model(input_model_file_name)
    nr_of_spectra = 2
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(
        model.encoder,
        (torch.rand(nr_of_spectra, model.model_settings.number_of_bins()),
         torch.rand(nr_of_spectra, len(model.model_settings.additional_metadata)),
         )
    )

    # Save the TorchScript model
    traced_script_module.save(output_model_file_name)


def save_model_for_java_predict_score(input_model_file_name,
                                      output_model_file_name):
    """Saves a model in a file format compatible with java
    The model will take in two spectra and predict the similarity score.
    """
    model = load_model(input_model_file_name)
    nr_of_spectra = 2
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(
        model,
        (torch.rand(nr_of_spectra, model.model_settings.number_of_bins()),
         torch.rand(nr_of_spectra, model.model_settings.number_of_bins()),
         torch.rand(nr_of_spectra, len(model.model_settings.additional_metadata)),
         torch.rand(nr_of_spectra, len(model.model_settings.additional_metadata))))

    # Save the TorchScript model
    traced_script_module.save(output_model_file_name)
