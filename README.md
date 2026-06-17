# whisper-live

Transcription microphone en temps réel via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) sur Apple Silicon.

Le texte s'affiche et se met à jour pendant que vous parlez (sliding window), puis se valide dès que vous faites une pause.

## Installation

```bash
pip install mlx-whisper sounddevice numpy
```

## Utilisation

```bash
python transcribe.py --language fr
```

Le modèle se télécharge automatiquement depuis HuggingFace au premier lancement.

## Modèles recommandés

Tous les modèles tournent sur le Neural Engine via MLX — même `large-v3` est utilisable en temps réel sur Apple Silicon.

| Modèle | Qualité | Vitesse | Pour |
|---|---|---|---|
| `mlx-community/whisper-tiny-mlx` | ★★☆☆☆ | ultra-rapide | tests, démos |
| `mlx-community/whisper-base-mlx` | ★★★☆☆ | très rapide | usage quotidien simple |
| `mlx-community/whisper-small-mlx` | ★★★★☆ | rapide | **défaut recommandé** |
| `mlx-community/distil-whisper-large-v3` | ★★★★☆ | rapide | meilleur ratio qualité/vitesse |
| `mlx-community/whisper-large-v3-turbo` | ★★★★★ | rapide | qualité maximale |
| `mlx-community/whisper-large-v3-mlx` | ★★★★★ | modéré | qualité maximale, accents difficiles |

### Par puce Apple Silicon

| Puce | Recommandation |
|---|---|
| M1 / M2 | `whisper-small-mlx` ou `distil-whisper-large-v3` |
| M3 / M4 | `distil-whisper-large-v3` ou `whisper-large-v3-turbo` |
| M5 | `whisper-large-v3-turbo` ou `whisper-large-v3-mlx` |

## Options

```
--model       Repo HuggingFace du modèle (défaut: mlx-community/whisper-small-mlx)
--language    Code langue ISO 639-1, ex: fr, en (défaut: auto-détection)
--threshold   Seuil RMS de détection de parole (défaut: 0.01)
--silence     Durée de silence avant finalisation en secondes (défaut: 0.8)
--interval    Intervalle de re-transcription pendant la parole en secondes (défaut: 1.0)
--timestamps  Afficher les timestamps de chaque segment
--list-devices  Lister les périphériques audio disponibles
```

## Exemples

```bash
# Français, modèle rapide
python transcribe.py --language fr

# Meilleure qualité sur M3+
python transcribe.py --model mlx-community/whisper-large-v3-turbo --language fr

# Réactivité maximale (silence plus court, mise à jour plus fréquente)
python transcribe.py --silence 0.5 --interval 0.8 --language fr

# Voir les périphériques audio disponibles
python transcribe.py --list-devices
```
