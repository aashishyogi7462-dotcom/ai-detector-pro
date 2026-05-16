from flask import Flask, render_template, request
import os
import cv2
from transformers import pipeline
from PIL import Image
import uuid

app = Flask(__name__)
@app.route("/features")
def features():
    return render_template("features.html")


@app.route("/about")
def about():
    return render_template("about.html")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 🔥 LOAD MODELS (ONLY ONCE)
image_detector = pipeline("image-classification", model="umm-maybe/AI-image-detector")

text_detector = pipeline(
    "text-classification",
    model="openai-community/roberta-base-openai-detector"
)
# -------- TEXT ANALYSIS --------
def analyze_text(text):
    words = text.lower().split()

    if len(words) < 10:
        return {"label": "⚠️ Text Too Short", "confidence": 0}

    # 🔹 AI style words
    ai_words = [
        "in conclusion", "moreover", "furthermore",
        "thus", "hence", "overall", "significantly"
    ]

    ai_count = sum(1 for w in ai_words if w in text.lower())

    # 🔹 repetition check
    unique_ratio = len(set(words)) / len(words)

    # 🔥 Decision logic
    if ai_count >= 2:
        return {"label": "🤖 AI Generated", "confidence": 80}

    elif unique_ratio < 0.5:
        return {"label": "⚠️ Repetitive / Copied", "confidence": 60}

    else:
        return {"label": "🧠 Human Written", "confidence": 75}
# -------- IMAGE --------
def analyze_image_model(path):
    try:
        img = Image.open(path)
        result = image_detector(img)[0]

        label = result.get("label", "").lower()
        score = result.get("score", 0)

        confidence = int(score * 100)

        # 🔥 Smart detection
        if "generated" in label or "ai" in label or "fake" in label:
            return {
                "label": "🤖 AI Image",
                "confidence": confidence
            }

        elif "real" in label or "photo" in label:
            return {
                "label": "🧠 Real Image",
                "confidence": confidence
            }

        else:
            return {
                "label": f"⚠️ {label}",
                "confidence": confidence
            }

    except Exception as e:
        print("Image Error:", e)
        return {
            "label": "❌ Error",
            "confidence": 0
        }

# -------- SIMPLE IMAGE (VIDEO FRAMES) --------
def analyze_image_simple(path):
    try:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = gray.mean()

        if variance < 45 or brightness < 50:
            return "AI"
        else:
            return "Real"

    except:
        return "Error"


# -------- EXTRACT FRAMES --------
def extract_frames(video_path, output_folder="static/frames"):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % 25 == 0:
            frame_path = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)

        if len(frames) >= 6:
            break

        count += 1

    cap.release()
    return frames


# -------- VIDEO --------
def analyze_video(path):
    try:
        frames = extract_frames(path)

        if not frames:
            return "🎥 Unable to analyze"

        ai_count = 0
        real_count = 0

        for frame in frames:
            result = analyze_image_simple(frame)

            if result == "AI":
                ai_count += 1
            elif result == "Real":
                real_count += 1

        total = len(frames)

        if ai_count > real_count:
            return f"🤖 AI Video ({ai_count}/{total})"
        elif real_count > ai_count:
            return f"🧠 Real Video ({real_count}/{total})"
        else:
            return f"⚠️ Uncertain ({total})"

    except Exception as e:
        return f"❌ Error: {str(e)}"


# -------- MAIN ROUTE --------
@app.route("/", methods=["GET", "POST"])
def home():
    text_result = None
    image_result = None
    video_result = None
    image_path = None
    video_path = None

    if request.method == "POST":

        # TEXT
        text = request.form.get("text")
        if text and text.strip():
            text_result = analyze_text(text)

        # IMAGE
        image = request.files.get("image")
        if image and image.filename != "":
            filename = str(uuid.uuid4()) + "_" + image.filename
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image.save(path)

            image_result = analyze_image_model(path)
            image_path = path

        # VIDEO
        video = request.files.get("video")
        if video and video.filename != "":
            filename = str(uuid.uuid4()) + "_" + video.filename
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            video.save(path)

            video_result = analyze_video(path)
            video_path = path

    return render_template(
        "index.html",
        text_result=text_result,
        image_result=image_result,
        video_result=video_result,
        image_path=image_path,
        video_path=video_path
    )


if __name__ == "__main__":
    app.run(debug=True)