import argparse, cv2, numpy as np, time, os, sys, subprocess, platform
from tensorflow import keras

# Speak-friendly mapping (avoid the ₹ symbol)
SPEAK_MAP = {
    "₹10":   "10 rupees",
    "₹20":   "20 rupees",
    "₹50":   "50 rupees",
    "₹100":  "100 rupees",
    "₹200":  "200 rupees",
    "₹500":  "500 rupees",
    "₹2000": "2000 rupees",
}
def speakable(label: str) -> str:
    return SPEAK_MAP.get(label, label.replace("₹", " rupees "))


# ---------- Speech helper (pyttsx3 with macOS 'say' fallback) ----------
class Speaker:
    def __init__(self, rate=170, voice_id=None, use_say_fallback=True):
        self.engine = None
        self.rate = rate
        self.voice_id = voice_id
        self.use_say_fallback = use_say_fallback and (platform.system() == "Darwin")
        try:
            import pyttsx3
            self.engine = pyttsx3.init()          # nsss driver on macOS
            self.engine.setProperty("rate", rate)
            if voice_id:
                self.engine.setProperty("voice", voice_id)
        except Exception:
            self.engine = None

    def say(self, text):
        if not text:
            return
        if self.engine is not None:
            try:
                self.engine.stop()
                self.engine.say(text)
                self.engine.runAndWait()
                return
            except Exception:
                pass
        if self.use_say_fallback:
            try:
                cmd = ["say"]
                if self.voice_id:
                    cmd += ["-v", self.voice_id]
                cmd += [text]
                subprocess.run(cmd, check=False)
            except Exception:
                pass
# ----------------------------------------------------------------------


# ---------- utils ----------
def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def preprocess_bgr(frame, img=224):
    x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (img, img), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32) / 255.0
    return np.expand_dims(x, 0)

def speak(engine, text):
    if not text: 
        return
    engine.stop()
    engine.say(text)
    engine.runAndWait()

def predict_keras(model, frame, img_size, labels):
    x = preprocess_bgr(frame, img_size)
    probs = model.predict(x, verbose=0)[0]
    i = int(np.argmax(probs))
    conf = float(probs[i])
    return labels[i], conf

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Desktop INR currency recognition (Keras + pyttsx3)")
    ap.add_argument("--model", required=True, help=".keras model path")
    ap.add_argument("--labels", default="labels.txt")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--every_n_frames", type=int, default=2)
    ap.add_argument("--speak_gap", type=float, default=1.5, help="seconds between repeated speech")
    ap.add_argument("--voice", default="", help="macOS voice id (e.g. com.apple.speech.synthesis.voice.samantha)")
    ap.add_argument("--rate", type=int, default=170)
    ap.add_argument("--image", type=str, default="", help="Run on a single image instead of webcam")

    args = ap.parse_args()

    labels = load_labels(args.labels)
    model = keras.models.load_model(args.model, compile=False)

    speaker = Speaker(rate=args.rate, voice_id=args.voice)

    # ---------- Single image-file mode ----------
    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(args.image)
        frame = cv2.imread(args.image)
        if frame is None:
            raise RuntimeError(f"Failed to read image: {args.image}")
        pred, conf = predict_keras(model, frame, args.img, labels)
        text = f"{pred} ({conf:.2f})"
        color = (0, 255, 0) if conf >= args.threshold else (0, 200, 255)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.imshow("Currency Recognition (Image)", frame)

        # Speak only if confident
        if conf >= args.threshold:
            speaker.say(speakable(pred))
        else:
            print(f"Low confidence: {text}")

        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ---------- Webcam mode ----------
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")

    last_spoke_t = 0.0
    last_confident_pred = ""   # for 'R' to repeat
    f = 0

    print("Controls:  ESC = quit   R = repeat last confident prediction")
    while True:
        ok, frame = cap.read()
        if not ok: break
        f += 1
        overlay_text = ""

        # Predict every Nth frame for speed
        if f % args.every_n_frames == 0:
            pred, conf = predict_keras(model, frame, args.img, labels)
            if conf >= args.threshold:
                overlay_text = f"{pred} ({conf:.2f})"
                now = time.time()
                if now - last_spoke_t > args.speak_gap:
                    speaker.say(speakable(pred))
                    last_spoke_t = now
                last_confident_pred = pred
            else:
                overlay_text = f"low conf ({conf:.2f})"

        if overlay_text:
            color = (0, 255, 0) if "low conf" not in overlay_text else (0, 200, 255)
            cv2.putText(frame, overlay_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        cv2.imshow("Currency Recognition (Webcam)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('r'), ord('R')):
            # Repeat the last confident label (if any)
            if last_confident_pred:
                speak(tts, last_confident_pred)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
