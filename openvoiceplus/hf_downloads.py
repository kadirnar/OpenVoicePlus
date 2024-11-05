from huggingface_hub import snapshot_download


def download_openvoice_model(model_version="v1"):
    """
    Download OpenVoice model based on version.

    Args:
        model_version (str): Version of the model ("v1" or "v2")
    """
    if model_version.lower() == "v1":
        repo_id = "myshell-ai/OpenVoice"
        local_dir = "./"
    elif model_version.lower() == "v2":
        repo_id = "myshell-ai/OpenVoiceV2"
        local_dir = "checkpoints_v2"
    else:
        raise ValueError("model_version must be either 'v1' or 'v2'")

    snapshot_download(
        repo_id=repo_id, repo_type="model", ignore_patterns=["*.md", "*..gitattributes"], local_dir=local_dir)
