import toml
import google.generativeai as genai
import sys

def test_model(name, api_key):
    print(f"Testing model: {name}...")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(name)
        response = model.generate_content("Ping")
        print(f"  Result: {response.text.strip()}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

config = toml.load("config.toml")
api_key = config['provider']['api_key']
models = config['provider']['models']

results = {}
for role, name in models.items():
    results[name] = test_model(name, api_key)

if all(results.values()):
    print("\nAll models are working correctly!")
    sys.exit(0)
else:
    print("\nSome models failed to respond.")
    sys.exit(1)
