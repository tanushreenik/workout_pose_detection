import cv2
import argparse
import os
from collections import defaultdict
import numpy as np

from pose_detector import PoseDetector
from posture_checker import PostureChecker
from pose_utils import smooth_time_series

# Optional MLFlow import
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLFlow not available. Install with: pip install mlflow")


class FormDetector:
    def __init__(self, exercise_type="bicep_curl", side="left", use_mlflow=False):
        self.pose_detector = PoseDetector()
        self.posture_checker = PostureChecker()
        self.exercise_type = exercise_type
        self.side = side
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE

        # Storage for time-series data
        self.metrics_history = defaultdict(list)
        self.frame_results = []

    def process_video(self, video_path, output_path=None, show_preview=True):
        # Open video
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {video_path}")
        print(
            f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Frames: {total_frames}"
        )

        # Setup video writer if output path specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Start MLFlow run
        if self.use_mlflow:
            mlflow.start_run()
            mlflow.log_param("video_path", video_path)
            mlflow.log_param("exercise_type", self.exercise_type)
            mlflow.log_param("side", self.side)
            mlflow.log_param("fps", fps)
            mlflow.log_param("total_frames", total_frames)

        frame_count = 0
        valid_frames = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Detect pose
                results, landmarks = self.pose_detector.detect(frame)

                # Draw pose landmarks
                frame = self.pose_detector.draw_landmarks(frame, results)

                # Check form if landmarks detected
                feedback_text = []
                if landmarks:
                    # Run posture checks
                    check_results = self.posture_checker.check_all(
                        landmarks, self.exercise_type, self.side
                    )

                    # Store results
                    self.frame_results.append(
                        {
                            "frame": frame_count,
                            "valid": check_results["overall_valid"],
                            "results": check_results,
                        }
                    )

                    if check_results["overall_valid"]:
                        valid_frames += 1

                    # Collect metrics for time-series
                    for check_name, check_data in check_results["checks"].items():
                        for metric_name, metric_value in check_data["metrics"].items():
                            key = f"{check_name}_{metric_name}"
                            self.metrics_history[key].append(metric_value)

                    # Get feedback
                    feedback_text = check_results["all_feedback"]
                else:
                    feedback_text = ["No pose detected"]

                # Draw feedback on frame
                self._draw_feedback(frame, feedback_text, frame_count, total_frames)

                # Write frame to output
                if out:
                    out.write(frame)

                # Show preview
                if show_preview:
                    cv2.imshow("Form Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Preview stopped by user")
                        break

                # Progress
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames...")

        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            self.pose_detector.close()

        # Calculate statistics
        accuracy = (valid_frames / frame_count * 100) if frame_count > 0 else 0

        # Smooth time-series data
        smoothed_metrics = {}
        for key, values in self.metrics_history.items():
            if len(values) > 5:
                smoothed_metrics[key] = smooth_time_series(values)
            else:
                smoothed_metrics[key] = values

        results_summary = {
            "video_path": video_path,
            "exercise_type": self.exercise_type,
            "side": self.side,
            "total_frames": frame_count,
            "valid_frames": valid_frames,
            "accuracy": accuracy,
            "smoothed_metrics": smoothed_metrics,
            "frame_results": self.frame_results,
        }

        # Log to MLFlow
        if self.use_mlflow:
            mlflow.log_metric("total_frames_processed", frame_count)
            mlflow.log_metric("valid_frames", valid_frames)
            mlflow.log_metric("accuracy_percent", accuracy)

            # Log average metrics
            for key, values in smoothed_metrics.items():
                if len(values) > 0:
                    mlflow.log_metric(f"avg_{key}", np.mean(values))
                    mlflow.log_metric(f"std_{key}", np.std(values))

            mlflow.end_run()

        print(f"\n{'='*50}")
        print(f"Analysis Complete!")
        print(f"Total Frames: {frame_count}")
        print(f"Valid Frames: {valid_frames}")
        print(f"Form Accuracy: {accuracy:.2f}%")
        print(f"{'='*50}\n")

        if output_path:
            print(f"Output video saved to: {output_path}")

        return results_summary

    def _draw_feedback(self, frame, feedback_text, frame_num, total_frames):
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (10, 10), (600, 150 + len(feedback_text) * 30), (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw title
        cv2.putText(
            frame,
            f"Exercise: {self.exercise_type.replace('_', ' ').title()} ({self.side})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Draw frame counter
        cv2.putText(
            frame,
            f"Frame: {frame_num}/{total_frames}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        # Draw feedback
        y_offset = 100
        for i, text in enumerate(feedback_text):
            color = (0, 255, 0) if "Good form" in text else (0, 165, 255)
            cv2.putText(
                frame,
                text,
                (20, y_offset + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )


def main():
    parser = argparse.ArgumentParser(description="Exercise Form Correctness Detection")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument(
        "--exercise",
        type=str,
        default="bicep_curl",
        choices=["bicep_curl", "lateral_raise"],
        help="Type of exercise",
    )
    parser.add_argument(
        "--side",
        type=str,
        default="left",
        choices=["left", "right"],
        help="Side to analyze",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save output video"
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="Disable real-time preview"
    )
    parser.add_argument("--mlflow", action="store_true", help="Enable MLFlow logging")

    args = parser.parse_args()

    # Validate video file
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return

    # Create output path if not specified
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = f"{base_name}_analyzed.mp4"

    # Create detector
    detector = FormDetector(
        exercise_type=args.exercise, side=args.side, use_mlflow=args.mlflow
    )

    # Process video
    try:
        results = detector.process_video(
            args.video, output_path=args.output, show_preview=not args.no_preview
        )

        print("\nDetection completed successfully!")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
