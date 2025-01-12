import requests
from bs4 import BeautifulSoup
import urllib.parse
import os

# Specify search term
search_term = "hiwonder raspberry pi robot car"
# Encode the search term to URL-friendly format
encoded_term = urllib.parse.quote_plus(search_term)

# Number of pages to iterate through
num_pages = 10
# Number of images per page (Google Images typically shows 20 images per page)
images_per_page = 20

# Create the save folder if it doesn't exist
save_folder = 'robot'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

results = []

for page in range(num_pages):
    # Construct Google Images query URL with pagination
    search_url = f"https://www.google.com/search?q={encoded_term}&tbm=isch&start={page * images_per_page}"

    # Request Google Images page
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    response = requests.get(search_url, headers=headers)

    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all img tags
    image_tags = soup.find_all('img')

    # Iterate over image tags
    for image in image_tags:
        # Get image source URL
        src = image.get('src')
        if src and src.startswith('http'):
            # Store data
            results.append(src)

# Save images with sequential filenames
for i, url in enumerate(results):
    file_name = f"{save_folder}/bin{i+1}.jpg"

    # Download image to folder
    try:
        urllib.request.urlretrieve(url, file_name)
        print(f"Downloaded {file_name}")
    except Exception as e:
        print(f"Could not download {url}. Error: {e}")