# 🍳 Let Em Cook

> Scan your receipt. Snap your fridge. Cook something great.

**Let Em Cook** is an AI-powered food waste reduction app built at the **Claude Builder Club Hackathon @ Imperial College London**. It turns your grocery receipt and whatever's in your fridge into a personalised meal plan — so nothing goes to waste.

## 👥 Team
Samuel Vasilis · Sophia Hussain · Tanvi Bhatla · Toni Disu

---

## 🧠 How It Works

1. **Scan your receipt** — upload a photo of your grocery receipt. Claude Vision extracts every item and quantity automatically.
2. **Scan fresh ingredients** — snap a photo of your fridge or produce. Claude identifies visible food items.
3. **Get meal suggestions** — the app suggests 4 meals ranked by how many of your ingredients they use, minimising waste.
4. **Identify a dish** — point the camera at any meal and the ML model names it instantly from 2,024 food classes.
5. **Get cooking guidance** — tap any meal for an AI-generated tip tailored to your exact ingredients.

---

## 🛠️ Tech Stack

| Layer | Tech |
|-------|------|
| Frontend | Vanilla HTML / CSS / JS — single-file mobile-first app |
| AI / Vision | Claude Sonnet (`claude-sonnet-4-20250514`) via Anthropic API |
| Food Classifier | TensorFlow SavedModel — AIY Food Vision (2,024 classes) |
| Backend | Python + Flask |
| Camera | WebRTC `getUserMedia` — live webcam capture in-browser |

---

## 📁 Project Structure
```
Let_em_Cook_final_.html         # Full frontend (single file)
server.py                        # Flask inference server
aiy-tensorflow1-vision-classifier-food-v1-v1/   # TF SavedModel
labels.txt                       # 2,024 food class labels
```

---

## 🚀 Running Locally

### 1. Start the Flask server
```bash
pip install flask tensorflow pillow numpy
python server.py
```

The server starts at `http://0.0.0.0:5001`.

### 2. Add your Anthropic API key

Open `Let_em_Cook_final_.html` and paste your key at line ~593:
```javascript
const ANTHROPIC_API_KEY = "sk-ant-...";
```

### 3. Open the app

Open `Let_em_Cook_final_.html` directly in your browser.  
Allow camera permissions when prompted.

---

## 🤖 Claude API Usage

Three separate Claude API calls power the experience:

**Receipt + ingredient scan** — multimodal prompt with image(s) returns structured JSON: receipt line items, detected fresh produce, and 4 ranked meal suggestions.

**Dish identification** — camera snapshot sent to Claude Vision; returns dish name, cuisine type, and confidence.

**Cooking guidance** — given a chosen meal and the user's ingredient list, Claude generates a short cooking tip highlighting what you have and what (if anything) to buy.

---

## ⚠️ Notes

- The Anthropic API key is currently set client-side — do not deploy publicly without moving this to a backend proxy.
- The Flask server must be running locally for the food classifier to work.
- The TF model directory must sit alongside `server.py`.

---

## 📄 License

Built for a hackathon. Open for learning and experimentation.
