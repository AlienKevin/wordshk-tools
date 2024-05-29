import sensenova

import csv

with open('sensenova_api_key.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        sensenova.access_key_id = row[0]
        sensenova.secret_access_key = row[1]

resp = sensenova.ChatCompletion.create(
    model="SenseChat-5",
    max_new_tokens=1024,
    messages=[
      {
        "role": "assistant",
        "content": "你係一個聰明、有趣嘅人工智能，名叫SenseChat，中文名係「商量」。你嘅回答需要條理清晰、邏輯清楚、內容詳細。你要用純正嘅粤語或英文回答用戶，而且盡量使用繁體字，咁樣先合符香港地區嘅閱讀習慣。"
      },
      {
        "role": "user",
        "content": "可以用IQ博士嘅角色，寫一篇日記嗎？"
      },
    ],
    repetition_penalty=1.05,
    temperature=0.8,
    top_p=0.7,
)

print(resp)
