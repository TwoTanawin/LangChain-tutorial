import requests
import json


def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = False):
    """scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile"""

def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = False):
    profileJson = "src/third_parties/eden-marco-scrapin.json"
    with open(profileJson) as f:
        d = json.load(f)
        # print(d)
        
    data = d.get("person")
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None) and k not in ["certifications"]
    }

    return data


if __name__ == "__main__":
    print(
        scrape_linkedin_profile(
            linkedin_profile_url="https://www.linkedin.com/in/eden-marco/"
        ),
    )