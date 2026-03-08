import requests

# นำ API Key ของคุณมาใส่ที่นี่
API_KEY = "AIzaSyBCENv_oCS4tJylW_w9O6xjxGP1U1-1wOs"
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

payload = {
    "contents": [{"parts": [{"text": "ทดสอบระบบ หากได้รับข้อความนี้ให้พิมพ์คำว่า OK"}]}]
}

print("กำลังยิงคำสั่งทดสอบ API Key ไปที่เซิร์ฟเวอร์ Google...")
response = requests.post(url, json=payload)

print(f"Status Code: {response.status_code}")
print("-" * 40)
print(response.text)