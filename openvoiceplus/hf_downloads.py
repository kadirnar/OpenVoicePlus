from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="myshell-ai/OpenVoice", 
    repo_type="model", 
    ignore_patterns=["*.md", "*..gitattributes"],
    local_dir="checkpoints",
    )
