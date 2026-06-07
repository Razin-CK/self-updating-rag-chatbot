import sys

print("Python version:", sys.version)

try:
    import google.generativeai as genai
    print("google.generativeai: SUCCESS, version:", genai.__version__)
except Exception as e:
    print("google.generativeai: FAILED,", str(e))

try:
    from google import genai as new_genai
    print("google.genai (new SDK): SUCCESS")
except Exception as e:
    print("google.genai (new SDK): FAILED,", str(e))

try:
    import openai
    print("openai: SUCCESS, version:", openai.__version__)
except Exception as e:
    print("openai: FAILED,", str(e))
