import cv2
import mediapipe as mp
import time

THEME = "dark"
themes = {
    "dark": {
        "normal": (40, 40, 40),       
        "hover": (0, 200, 255),       
        "text": (255, 255, 255)      
    },
    "light": {
        "normal": (220, 220, 220),    
        "hover": (0, 120, 255),       
        "text": (0, 0, 0)             
    },
    "neon": {
        "normal": (0, 0, 0),          
        "hover": (0, 255, 0),         
        "text": (0, 255, 255)         
    },
    "glass": {
        "normal": (100, 100, 100),    
        "hover": (255, 255, 255),     
        "text": (255, 255, 255)       
    }
}

theme_colors = themes[THEME]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"],
    ["SPACE", "ENTER", "CLEAR"]
]

def draw_rounded_rect(img, pt1, pt2, color, alpha=0.4):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1, cv2.LINE_AA)
    if THEME == "glass":
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        cv2.rectangle(img, pt1, pt2, color, -1, cv2.LINE_AA)
    cv2.rectangle(img, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)

def draw_keyboard(img, hover_key=None):
    key_positions = {}
    base_w, key_h = 80, 80
    start_x, start_y = 50, 120

    for row_idx, row in enumerate(keys):
        y = start_y + row_idx * (key_h + 15)
        x = start_x
        for key in row:
            if key == "SPACE":
                w = base_w * 4
            elif key in ["ENTER", "CLEAR"]:
                w = base_w * 2
            else:
                w = base_w

            rect_color = theme_colors["hover"] if key == hover_key else theme_colors["normal"]
            draw_rounded_rect(img, (x, y), (x + w, y + key_h), rect_color)

            # Center text
            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (key_h + text_size[1]) // 2
            cv2.putText(img, key, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, theme_colors["text"], 2)

            key_positions[key] = (x, y, w, key_h)
            x += w + 10
    return img, key_positions

def check_key(x, y, key_positions):
    for key, (kx, ky, kw, kh) in key_positions.items():
        if kx < x < kx + kw and ky < y < ky + kh:
            return key
    return None

cap = cv2.VideoCapture(0)
typed_text = ""
hover_key = None
key_start_time = 0
selected_key = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    frame, key_positions = draw_keyboard(frame, hover_key)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x = int(hand_landmarks.landmark[8].x * frame.shape[1])
            y = int(hand_landmarks.landmark[8].y * frame.shape[0])
            cv2.circle(frame, (x, y), 12, (0, 255, 0), -1)

            hover_key = check_key(x, y, key_positions)
            if hover_key:
                if selected_key != hover_key:
                    selected_key = hover_key
                    key_start_time = time.time()
                else:
                    if time.time() - key_start_time > 0.4:
                        if hover_key == "SPACE":
                            typed_text += " "
                        elif hover_key == "ENTER":
                            typed_text += "\n"
                        elif hover_key == "CLEAR":
                            typed_text = ""
                        else:
                            typed_text += hover_key
                        key_start_time = time.time() + 1
            else:
                selected_key = None

    cv2.rectangle(frame, (50, 30), (900, 90), (0, 0, 0), -1)
    cv2.putText(frame, typed_text[-40:], (60, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Air Keyboard", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

