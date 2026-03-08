import requests

# นำ API Key ที่เทสต์ผ่านเมื่อกี้มาใส่
API_KEY = "AIzaSyBCENv_oCS4tJylW_w9O6xjxGP1U1-1wOs"
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

print("กำลังค้นหาโมเดล AI ของ Google ที่คุณใช้งานได้...")
response = requests.get(url)
data = response.json()

if "models" in data:
    print("\n✅ รายชื่อโมเดลที่คุณสามารถเอาไปใส่ในโค้ดได้ (เลือกมา 1 อัน):")
    print("-" * 50)
    for model in data["models"]:
        # กรองเอาเฉพาะโมเดลที่ใช้สร้างข้อความได้
        if "generateContent" in model.get("supportedGenerationMethods", []):
            print(model["name"].replace("models/", ""))
    print("-" * 50)
else:
    print("เกิดข้อผิดพลาด:", data)