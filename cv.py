import cv2
from deepface import DeepFace
import threading
import time

class Main:
    "Класс создан для распознования лица человека \nACTIVATED✅"
    def main(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(3, 640)
        cap.set(4, 480)

        face_cascade = cv2.CascadeClassifier('face_model.xml')

        while True:
            start_time = time.time()
            suc, frame = cap.read()
            if not suc:
                break

            end_time = time.time()
            frame_number = 1 / (end_time - start_time)  # вычисление FPS по стандартной формуле 1/(конец-начало)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"FPS: {frame_number:.2f}", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 1)
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print("FINIS✅")

        cap.release()
        cv2.destroyAllWindows()


Mainer = Main()
print(Main.__doc__)

class Analyze:
    "Класс создан для распознования эмоционально-физиологических характеристик человека \nACTIVATED✅"
    def analyze_face(self, frame):
        try:
            self.result = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'])
            return self.result
        except Exception as ex:
            return str(ex)

analyze__face = Analyze()
print(Analyze.__doc__)

def video():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(4, 1080)
    cap.set(3, 1920)

    current_analysis = None
    analyzing = False

    def analyze_face_async(face_image):
        nonlocal current_analysis, analyzing
        try:
            result = analyze__face.analyze_face(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            current_analysis = result
        finally:
            analyzing = False

    face_cascade = cv2.CascadeClassifier('face_model.xml')

    while True:
        start_time = time.time()
        suc, frame = cap.read()
        if not suc:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]

            if not analyzing:
                analyzing = True
                thread = threading.Thread(target=analyze_face_async, args=(face_roi,))
                thread.daemon = True
                thread.start()

            if current_analysis and isinstance(current_analysis, list):
                for face_data in current_analysis:
                    age = face_data.get('age')
                    gender = face_data.get('gender')
                    emotion = face_data.get('dominant_emotion')
                    y_offset = y - 10
                    cv2.putText(frame, f"Age: {int(age)}", (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2)
                    y_offset -= 25
                    woman = gender.get('Woman')
                    man = gender.get('Man')
                    cv2.putText(frame, f"Gender: Woman: {round(woman, 2)}%, Man: {round(man, 2)}%", (x, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2)
                    y_offset -= 25

                    cv2.putText(frame, f"Emotion: {emotion}", (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2)
                    y_offset -= 25

        end_time = time.time()
        frame_number = 1 / (end_time - start_time)


        cv2.putText(frame, f"FPS: {frame_number:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('f'):
            break
    print("FINIS✅")

    cap.release()
    cv2.destroyAllWindows()

Mainer.main()
video()
