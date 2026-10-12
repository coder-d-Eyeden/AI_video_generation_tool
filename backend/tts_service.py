"""
Bengali TTS (Text-to-Speech) Service.
Synthesizes everyday Bengali narration scripts into playable audio for the preview player.
"""
import io
import base64
from gtts import gTTS


class TTSService:
    @staticmethod
    def synthesize_bengali_voiceover(
        bengali_text: str,
        start_pad_sec: float = 1.0,
        end_pad_sec: float = 1.5
    ) -> dict:
        """
        Synthesizes the Bengali text to MP3 audio using Bengali dialect voice.
        Returns a dict with audio_base64, estimated_speech_seconds, and word_count.
        """
        clean_text = (bengali_text or "").strip()
        if not clean_text:
            raise ValueError("No Bengali narration text provided for synthesis.")

        words = clean_text.split()
        word_count = len(words)

        # Average spoken Bengali tempo is ~2.8 to 3.2 syllables or words per second
        estimated_duration = max(3.0, round(word_count / 2.7, 1))

        mp3_fp = io.BytesIO()
        tts = gTTS(text=clean_text, lang='bn', slow=False)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_bytes = mp3_fp.read()

        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        data_uri = f"data:audio/mp3;base64,{b64_audio}"

        return {
            "audio_data_uri": data_uri,
            "word_count": word_count,
            "estimated_duration_sec": estimated_duration,
            "timing_compliance": {
                "narration_start_sec": start_pad_sec,
                "narration_end_sec": start_pad_sec + estimated_duration,
                "fits_within_10s_scene": (start_pad_sec + estimated_duration) <= 8.5
            }
        }

    @staticmethod
    def synthesize_to_file(bengali_text: str, out_path: str) -> str:
        """Writes Bengali narration to an MP3 file (used when mixing voiceover into rendered video)."""
        clean_text = (bengali_text or "").strip()
        if not clean_text:
            raise ValueError("No Bengali narration text provided for synthesis.")
        gTTS(text=clean_text, lang='bn', slow=False).save(out_path)
        return out_path
