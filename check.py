"""Checks each free service the app uses and says which one is slow or failing.

Run from the project folder:   python check.py
Each result is printed as soon as it's ready. It uses the keys in your .env file and never prints them.
Press Ctrl + C at any time to stop.
"""
from scout.config import Settings
from scout.health import iter_checks


def main() -> None:
    s = Settings.from_env()
    print(f"\nKeys found: Gemini {'yes' if s.gemini_key else 'no'}, Groq {'yes' if s.groq_key else 'no'}, "
          f"Claude {'yes' if s.claude_key else 'no (optional, paid)'}, "
          f"Jina {'yes' if s.jina_key else 'no (optional)'}, "
          f"Google Maps {'yes' if s.google_maps_key else 'no (optional)'} | main AI: {s.main_ai}")
    if not s.has_ai:
        print("No AI key found. Run:  open -e .env   and paste your key after GEMINI_API_KEY=  then save.")
    print("Checking each service (each takes up to about 30 seconds)...\n")
    for name, run in iter_checks(s):
        print(f"   checking {name} ...", flush=True)
        c = run()
        print(f"{'OK    ' if c.ok else 'FAILED'} {c.service:<32} {c.seconds:>5}s  {c.detail}")
    print("\nDone. If anything says FAILED, send a screenshot of this.\n")


if __name__ == "__main__":
    main()
