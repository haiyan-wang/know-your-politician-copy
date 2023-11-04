from bs4 import BeautifulSoup
import requests


# Returns urls from Reuter's front page news
def retrieve_reuters(article):
    try:
        html = requests.get(f'https://www.govinfo.gov/content/pkg/CHRG-118shrg{article}/html/CHRG-118shrg{article}.htm').text
        soup = BeautifulSoup(html, 'html.parser')

        if soup.text.strip()[:14] == "Page Not Found":
            return

        with open(f'senate_hearings/{article}.txt', 'w') as file:
            file.write(soup.text)

        file.close()
    except:
        return

# for num in range(54000, 55000):
#     if num % 50 == 0:
#         print(num)
#     retrieve_reuters(num)

retrieve_reuters(49104033)