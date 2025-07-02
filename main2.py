import cv2
import numpy as np
import streamlit as st
from tempfile import NamedTemporaryFile

class WeedDetection:
    def __init__(self, img):
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.part_width = img.shape[1] // 3

    def preprocess(self, img):
        kernel_size = 15
        img_blur = cv2.medianBlur(img, kernel_size)
        img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        return img_hsv

    def createMask(self, img_hsv):
        sensitivity = 20
        lower_bound = np.array([50 - sensitivity, 100, 60])
        upper_bound = np.array([50 + sensitivity, 255, 255])
        msk = cv2.inRange(img_hsv, lower_bound, upper_bound)
        return msk

    def transform(self, msk):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        res_msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, kernel)
        res_msk = cv2.morphologyEx(res_msk, cv2.MORPH_CLOSE, kernel)
        return res_msk

    def calcPercentage(self, msk):
        height, width = msk.shape[:2]
        num_pixels = height * width
        count_white = cv2.countNonZero(msk)
        percent_white = (count_white / num_pixels) * 100
        return round(percent_white, 2)

    def weedPercentage(self, msk):
        left_part = msk[:, :self.part_width]
        mid_part = msk[:, self.part_width:2 * self.part_width]
        right_part = msk[:, 2 * self.part_width:]
        left_percent = self.calcPercentage(left_part)
        mid_percent = self.calcPercentage(mid_part)
        right_percent = self.calcPercentage(right_part)
        return [left_percent, mid_percent, right_percent]

    def markPercentage(self, img, percentage):
        part_width = self.width // 3
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(3):
            cv2.putText(img, f"{percentage[i]}%", (int(part_width * (i + 0.34)), self.height // 2), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
        return img

# Streamlit App
st.title("Weed Detection Tool")
st.write("Upload an image or a video to detect weeds and their distribution.")

media_type = st.selectbox("Select Media Type", ["Image", "Video"])
uploaded_file = st.file_uploader("Upload a file", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file:
    if media_type == "Image":
        # Process Image
        image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if img is not None:
            wd = WeedDetection(img)
            img_hsv = wd.preprocess(img)
            msk = wd.createMask(img_hsv)
            msk = wd.transform(msk)
            percentage = wd.weedPercentage(msk)
            res = wd.markPercentage(img.copy(), percentage)
            
            st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), caption="Processed Image")
            st.write(f"Weed Percentages (Left, Middle, Right): {percentage}")
        else:
            st.error("Could not process the uploaded image.")

    elif media_type == "Video":
        # Process Video
        temp_file = NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()

        if ret:
            wd = WeedDetection(first_frame)

            frame_window = st.empty()
            percentage_window = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_hsv = wd.preprocess(frame)
                msk = wd.createMask(frame_hsv)
                msk = wd.transform(msk)
                percentage = wd.weedPercentage(msk)
                res = wd.markPercentage(frame.copy(), percentage)

                frame_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, channels="RGB")
                percentage_window.write(f"Weed Percentages (Left, Middle, Right): {percentage}")

                if cv2.waitKey(2) & 0xFF == 27:
                    break

            cap.release()
        else:
            st.error("Could not process the uploaded video.")

    else:
        st.error("Unsupported media type.")
