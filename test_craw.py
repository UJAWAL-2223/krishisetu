import requests
from xml.etree import ElementTree

NS = "http://www.sitemaps.org/schemas/sitemap/0.9"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

# Step 1: fetch sitemap index
r = requests.get("https://www.myscheme.gov.in/sitemap.xml", headers=headers)
root = ElementTree.fromstring(r.content)

print("Tags found in sitemap index:")
for child in root:
    print(" ", child.tag, "→", child.text or "")
    for subchild in child:
        print("    ", subchild.tag, "→", subchild.text or "")

# Step 2: fetch sitemap-0.xml
r2 = requests.get("https://www.myscheme.gov.in/sitemap-0.xml", headers=headers)
root2 = ElementTree.fromstring(r2.content)

print("\nFirst 5 tags in sitemap-0.xml:")
for i, child in enumerate(root2):
    if i >= 5:
        break
    print(" ", child.tag)
    for subchild in child:
        print("    ", subchild.tag, "→", subchild.text or "")

print("\nTotal entries in sitemap-0.xml:", len(list(root2)))