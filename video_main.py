# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Class with methods to do emotion analysis
on video or webcam feed.
Usage: python video_main"""
# ===================================================
import os  # Added missing import for os module
import sys
import time
from typing import Dict, List

import cv2
import numpy as np
from emotion_analyzer.emotion_detector import EmotionDetector
from emotion_analyzer.logger import LoggerFactory
from emotion_analyzer.media_utils import (
    annotate_emotion_stats,
    annotate_warning,
    convert_to_rgb,
    draw_bounding_box_annotation,
    draw_emoji,
    get_video_writer,
)
from emotion_analyzer.validators import path_exists

# Load the custom logger
logger = None
try:
    logger_ob = LoggerFactory(logger_name=__name__)
    logger = logger_ob.get_logger()
    logger.info("{} loaded...".format(__name__))
    # Set exception hook for uncaught exceptions
    sys.excepthook = logger_ob.uncaught_exception_hook
except Exception as exc:
    raise exc


class EmotionAnalysisVideo:
    """Class with methods to do emotion analysis on video or webcam feed."""
    emoji_foldername = "emojis"

    def __init__(
        self,
        face_detector: str = "dlib",
        model_loc: str = "models",
        face_detection_threshold: float = 0.8,
        emoji_loc: str = "data",
    ) -> None:
        # Construct the path to emoji folder
        self.emoji_path = os.path.join(emoji_loc, EmotionAnalysisVideo.emoji_foldername)
        # Load the emojis
        self.emojis = self.load_emojis(emoji_path=self.emoji_path)
        self.emotion_detector = EmotionDetector(
            model_loc=model_loc,
            face_detection_threshold=face_detection_threshold,
            face_detector=face_detector,
        )

    def emotion_analysis_video(
        self,
        video_path: str = None,
        detection_interval: int = 15,
        save_output: bool = False,
        preview: bool = False,
        output_path: str = "data/output.mp4",
        resize_scale: float = 0.5,
    ) -> None:
        # If no video source is given, try switching to webcam
        video_path = 0 if video_path is None else video_path
        if not path_exists(video_path) and video_path != 0:  # Allow webcam (video_path == 0)
            raise FileNotFoundError(f"The video file '{video_path}' was not found.")

        cap, video_writer = None, None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video source.")

            # To save the video file, get the OpenCV video writer
            video_writer = get_video_writer(cap, output_path) if save_output else None

            frame_num = 1
            t1 = time.time()
            logger.info("Enter 'q' to exit...")
            emotions = None

            while True:
                status, frame = cap.read()
                if not status:
                    break

                try:
                    # Flip webcam feed so that it looks mirrored
                    if video_path == 0:
                        frame = cv2.flip(frame, 1)  # Corrected flip code from 2 to 1

                    if frame_num % detection_interval == 0:
                        # Scale down the image to increase model inference speed
                        smaller_frame = convert_to_rgb(
                            cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                        )
                        # Detect emotion
                        emotions = self.emotion_detector.detect_emotion(smaller_frame)

                    # Annotate the current frame with emotion detection data
                    frame = self.annotate_emotion_data(emotions, frame, resize_scale)

                    if save_output:
                        video_writer.write(frame)

                    if preview:
                        cv2.imshow("Preview", cv2.resize(frame, (680, 480)))

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                except Exception as exc:
                    logger.error(f"Error processing frame {frame_num}: {exc}")
                    raise exc

                frame_num += 1

            t2 = time.time()
            logger.info("Time: {:.2f} minutes".format((t2 - t1) / 60))
            logger.info("Total frames: {}".format(frame_num))
            logger.info("Time per frame: {:.4f} seconds".format((t2 - t1) / frame_num))

        except Exception as exc:
            logger.critical(f"Uncaught Exception raised! {exc}")
            raise exc

        finally:
            cv2.destroyAllWindows()
            if cap is not None:
                cap.release()
            if video_writer is not None:
                video_writer.release()

    def load_emojis(self, emoji_path: str = "data/emojis") -> Dict[str, np.ndarray]:
        """Load emoji images corresponding to different emotions."""
        emojis = {}
        # List of given emotions
        EMOTIONS = [
            "Angry",
            "Disgusted",
            "Fearful",
            "Happy",
            "Sad",
            "Surprised",
            "Neutral",
        ]
        # Store the emoji corresponding to different emotions
        for emotion in EMOTIONS:
            emoji_file_path = os.path.join(emoji_path, f"{emotion.lower()}.png")
            if not os.path.exists(emoji_file_path):
                logger.warning(f"Emoji file not found: {emoji_file_path}")
                continue
            emojis[emotion] = cv2.imread(emoji_file_path, -1)

        logger.info("Finished loading emojis...")
        return emojis

    def annotate_emotion_data(
        self, emotion_data: List[Dict], image, resize_scale: float
    ) -> np.ndarray:
        """Annotate the frame with emotion data."""
        # Draw bounding boxes for each detected person
        for data in emotion_data:
            image = draw_bounding_box_annotation(
                image,
                data["emotion"],
                (np.array(data["bbox"]) / resize_scale).astype(int),
            )

        # If there are more than one person in the frame, show a warning.
        WARNING_TEXT = "Warning! More than one person detected!"
        if len(emotion_data) > 1:
            image = annotate_warning(WARNING_TEXT, image)

        if len(emotion_data) > 0:
            # Draw emotion confidence stats
            image = annotate_emotion_stats(emotion_data[0]["confidence_scores"], image)
            # Draw the emoji corresponding to the emotion
            image = draw_emoji(self.emojis[emotion_data[0]["emotion"]], image)

        return image


if __name__ == "__main__":
    # SAMPLE USAGE
    ob = EmotionAnalysisVideo(
        face_detector="dlib",
        model_loc="models",
        face_detection_threshold=0.0,
    )

    # Use webcam as the video source
    video_path = 0  # Webcam input
    output_path = "data/output.mp4"
    save_output = True
    preview = True

    ob.emotion_analysis_video(
        video_path=video_path,
        detection_interval=1,
        save_output=save_output,
        preview=preview,
        output_path=output_path,
        resize_scale=0.5,
    )