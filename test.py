import os
import torch
from openvoiceplus import se_extractor
from openvoiceplus.api import ToneColorConverter
from meloplus.api import TTS
from openvoiceplus.hf_downloads import download_openvoice_model

download_openvoice_model(model_version="v2")


def load_model(ckpt_path='checkpoints_v2/converter', output_dir='outputs_v2'):
    """
    Loads the model and required components

    Args:
        ckpt_path: Model checkpoint directory
        output_dir: Output directory

    Returns:
        tone_color_converter, device, target_se
    """
    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create directory
    os.makedirs(output_dir, exist_ok=True)

    # Load tone color converter
    tone_color_converter = ToneColorConverter(f'{ckpt_path}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_path}/checkpoint.pth')

    # Extract speaker embedding from reference audio
    reference_speaker = 'resources/example_reference.mp3'
    target_se, _ = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)

    return tone_color_converter, device, target_se


def generate_audio(tone_color_converter, device, target_se, output_dir='outputs_v2', speed=1.0):
    """
    Generates audio in different languages

    Args:
        tone_color_converter: Loaded converter model
        device: Device to use (cpu/cuda)
        target_se: Target speaker embedding
        output_dir: Output directory
        speed: Speech speed
    """
    # Sample texts
    texts = {
        'EN_NEWEST': "Did you ever hear a folk tale about a giant turtle?",
        'EN': "Did you ever hear a folk tale about a giant turtle?",
        'ES': "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante.",
        'FR': "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
        'ZH': "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。",
        'JP': "彼は毎朝ジョギングをして体を健康に保っています。",
        'KR': "안녕하세요! 오늘은 날씨가 정말 좋네요.",
    }

    src_path = f'{output_dir}/tmp.wav'

    for language, text in texts.items():
        model = TTS(language=language, device=device)
        speaker_ids = model.hps.data.spk2id

        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            speaker_key = speaker_key.lower().replace('_', '-')

            # Load source speaker embedding
            source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
            save_path = f'{output_dir}/output_v2_{speaker_key}.wav'

            # Generate audio with TTS
            model.tts_to_file(text, speaker_id, src_path, speed=speed)

            # Apply tone color conversion
            tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=save_path,
                message="@MyShell")


def main():
    # Load model
    converter, device, target_se = load_model(ckpt_path='checkpoints_v2/converter', output_dir='outputs_v2')

    # Generate audio
    generate_audio(converter, device, target_se)


if __name__ == "__main__":
    main()
