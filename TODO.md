
use deepgram speech synth as the test input stt in keepalive_stability.py


=====

Add openai as llm provider



==========
- Implement macos hear command as a stt option

documentation:
```
HEAR(1)                     General Commands Manual                    HEAR(1)

NAME

     hear – macOS speech recognition and transcription via the command line

SYNOPSIS

     hear [-hvdmpsTSa] [-i audio_file] [-l locale] [-x exit_word]
          [-t timeout_sec] [-n input_device_id]

DESCRIPTION

     hear is a command line interface for the built-in speech recognition
     capabilities in macOS. It supports transcription of both audio files and
     data from audio input devices. If no input file is provided, the default
     behaviour is to transcribe from the default audio input device, typically
     the microphone.

     The following flags are supported:

     -s --supported
              Print list of supported locales for speech recognition and exit.

     -l --locale loc
              Specify speech recognition language (locale). Default is 'en-
              US'.

     -i --input audio_file
              Input audio file. All formats supported by CoreAudio should work
              (e.g. WAV, MP3, AAC, CAF, AIFF, ALAC, etc.).

     -d --device
              Only use on-device offline speech recognition. The default is to
              use whatever the macOS Speech Recognition API thinks is best,
              which may include sending data to Apple servers. When on-device
              is not enabled, there may be a hard limit to the length of audio
              that can be transcribed in a single session. As of writing
              (2025) this seems to be about 500 characters or so.

     -m --mode
              Enable single-line output mode (only applies when the input an
              audio input device, e.g. a microphone).

     -p --punctuation
              Add punctuation to speech recognition results.

     -T --timestamps
              Write timestamps as transcription occurs (file input only).

     -S --subtitle
              Enable subtitle mode, producing .srt output (file input only,
              experimental)

     -x --exit exit_word
              Set exit word. This causes the program to exit when a speech
              recognition result ends with the specified word.

     -t --timeout seconds
              Exit if no recognition results are received within the specified
              number of seconds.

     -a --audio-input-devices
              List available audio input devices and exit.

     -n ---input-device-id
              Specify ID of audio input device (experimental).

     -h --help
              Print help and exit.

     -v --version
              Print program name and version, and exit.

     Returns 0 on success and greater than 0 on error or failure.


```
Make sure that interruption is working correctly; when stt detects an utterance and interrupts an ongoing tts utterance, it should cut off audio output cleanly and in a safe manner. the following should be implemented in the tts implementations but I mention it here so that you implement messaging to it correctly; only words that are actually uttered should be added to the transcript log of the conversation.

use -d --device for local inference, and -p --punctuation to add punctuation to the transcript.

the hear command outputs interim results as well as final results. make sure that interim results are handled correctly; they should not be added to the conversation history, only the final result should be. 

output looks like this:

Hello
Hello can
Hello can U
Hello can you
Hello can you hear
Hello can you hear me
Hello can you hear me how
Hello can you hear me how are
Hello can you hear me how are you
Hello can you hear me how are you doing to
Hello can you hear me how are you doing today. 

It will always append to the previous output. It could be smart to restart the hear process after a final result is received and send to the llm, to avoid having an overflowing state. Make sure to restart immediately after final result is received to avoid missing any audio input, 

Sometimes it makes mistakes and corrects them later, so make sure that only the final correct text is added to the conversation history. to do this have a timeout parameter that waits for a 1.2 seconds of no new output before considering the last output as final.

You are currently running on a mac os device, so test if everything works, before finishing iteration
```

=============

- implement ollama as a llm provider. 
Implement the ollama api as a functioning llm provider. Ollama is runnable on this machine. In the config models should be specfified as ollama-{MODEL_ID}. gemma3:4b and qwen3-vl are a model IDs that are installed and you can test inference with. gemma does not start with a reasoning trace, while qwen does. Make sure your implementations works with models that output reasoning traces (these should never be sent to the tts or conversation history, or be parsed for <commands>).

Make sure that interruption is working correctly; when stt detects an utterance and interrupts an ongoing generation, it should be stopped in a safe manner. 
