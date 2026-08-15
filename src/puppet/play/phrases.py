"""Canned kid phrases for play start/stop (Piper; no LLM)."""

from __future__ import annotations

_PHRASES: dict[str, dict[str, str]] = {
  "follow": {
    "en": "Okay! I will follow you. Say stop if you want me to stay.",
    "fr": "D'accord! Je te suis. Dis arrete si tu veux que je reste.",
    "de": "Okay! Ich folge dir. Sag Stopp wenn ich stehen bleiben soll.",
  },
  "seek": {
    "en": "One, two, three, four, five, six, seven, eight, nine, ten. Ready or not, here I come!",
    "fr": "Un, deux, trois, quatre, cinq, six, sept, huit, neuf, dix. J'arrive!",
    "de": "Eins, zwei, drei, vier, fünf, sechs, sieben, acht, neun, zehn. Ich komme!",
  },
  "found": {
    "en": "Found you!",
    "fr": "Te voila ! Je t'ai trouve !",
    "de": "Gefunden! Da bist du!",
  },
  "giveup": {
    "en": "I cannot find you. Let's stop.",
    "fr": "Je ne te trouve pas. On arrete ?",
    "de": "Ich finde dich nicht. Wir hoeren auf.",
  },
  "closer": {
    "en": "I'm coming closer!",
    "fr": "J'arrive, je me rapproche!",
    "de": "Ich komme naeher!",
  },
  "idle": {
    "en": "Okay, I will stay here.",
    "fr": "D'accord, je reste ici.",
    "de": "Okay, ich bleibe hier.",
  },
  "back": {
    "en": "Okay, I will reverse a little.",
    "fr": "D'accord, je recule un peu.",
    "de": "Okay, ich fahre ein Stueck zurueck.",
  },
  "disabled": {
    "en": "I would love to roll, but moving is switched off right now.",
    "fr": "J'aimerais bien rouler, mais le mouvement est coupe pour l'instant.",
    "de": "Ich wuerde gern fahren, aber Bewegung ist gerade aus.",
  },
  "no_drive": {
    "en": "I cannot roll yet. The wheels are not ready.",
    "fr": "Je ne peux pas encore rouler. Les roues ne sont pas pretes.",
    "de": "Ich kann noch nicht fahren. Die Raeder sind nicht bereit.",
  },
}


def play_phrase(kind: str, lang: str = "en") -> str:
  lang = (lang or "en").lower()[:2]
  row = _PHRASES.get(kind) or _PHRASES["idle"]
  return row.get(lang) or row["en"]
