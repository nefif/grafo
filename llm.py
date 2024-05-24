import google.cloud.natural_language as natural_language

# Initialize Natural Language API client
client = natural_language.Client()

# Load text from file
with open("document.txt", "r") as f:
    text = f.read()

# Send text to Natural Language API for keyword extraction
response = client.analyze_text(text_content=text, encoding="UTF-8")

# Process API response
extracted_keywords = []
for keyword in response.keywords:
    extracted_keywords.append((keyword.text, keyword.saliency))

# Refine and analyze keywords (optional)
# ...

# Sort keywords by saliency score
sorted_keywords = sorted(extracted_keywords, key=lambda x: x[1], reverse=True)

# Select top 10 keywords
top_10_keywords = sorted_keywords[:10]

# Print top 10 keywords
print("Top 10 Keywords:")
for keyword, saliency in top_10_keywords:
    print(f"Keyword: {keyword}, Saliency: {saliency}")
