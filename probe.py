from audio_dsp import load_audio, analyze, ControlParams

y = load_audio("/mnt/c/Users/karti/Downloads/yourtrack.mp3")
ft = analyze(y, 44100, 60)

for s in (0.3, 0.5, 0.8):
    strength, _ = ft.onsets(s)
    print(f"sensitivity {s}: {int((strength > 0).sum())} onsets")

print(ft.frame(3000, ControlParams(onset_sensitivity=0.3)))
