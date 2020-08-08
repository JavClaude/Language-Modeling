import requests


def get_fake_financial_news(n_try: int = 10):

    i = 0

    while i < n_try:
        seed_str = input("Type something about financial content..\n")

        if seed_str == "stop":
            break

        json_data = {
            'text': seed_str
        }

        fake_financial_news = requests.post(
            url="http://127.0.0.1:5000/generate",
            json=json_data
            )
        
        print("---- Fake Generated News ----\n{}".format(fake_financial_news.text))
        
        i+=1

if __name__ == "__main__":
    get_fake_financial_news()
